"""Length-prefixed TCP messaging utilities.

This module provides a small networking layer used by the MOT control and ML
interfaces. Messages are framed as:
- 8-byte ASCII length header (zero padded), followed by
- UTF-8 payload bytes.
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import List, Optional, Tuple

_LOGGER = logging.getLogger("NetworkService")

PREFIX_BYTES = 8
RECV_CHUNK_SIZE = 1024
SYNC_MARKER = "<CC>"


class SocketWrapper:
    """Thin wrapper around a TCP socket with reusable defaults."""

    def __init__(self, host: Optional[str] = None, service_port: Optional[int] = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (OSError, AttributeError):
            _LOGGER.debug("SO_REUSEPORT is not available on this platform")

        self.host = host
        self.service_port = service_port

    def set_timeout(self, timeout_sec: float):
        """Set socket timeout in seconds."""
        self.sock.settimeout(timeout_sec)

    def listen(self):
        """Start listening for incoming connections."""
        self.sock.listen()

    def bind(self):
        """Bind socket to configured address/port."""
        if self.host is None or self.service_port is None:
            _LOGGER.error(
                "Socket configured without address/port: (%s:%s)",
                self.host,
                self.service_port,
            )
            raise ValueError("Socket address and port must be set before bind()")
        self.sock.bind((self.host, self.service_port))

    def accept_connections(self):
        """Accept one incoming connection."""
        return self.sock.accept()

    def connect(self):
        """Connect socket to configured address/port."""
        if self.host is None or self.service_port is None:
            raise ValueError("Socket address and port must be set before connect()")
        _LOGGER.info("Connecting to (%s,%s)", self.host, self.service_port)
        self.sock.connect((self.host, self.service_port))

    def read(self):
        """Read up to default buffer size and decode UTF-8."""
        return self.sock.recv(RECV_CHUNK_SIZE).decode("utf-8")

    def send(self, data):
        """Send raw bytes payload."""
        self.sock.send(data)


# Backward compatibility alias
TCPSocket = SocketWrapper


class PeerConnection:
    """Represents one framed-message TCP peer connection."""

    def __init__(self, peer_sock: socket.socket, peer_addr: Tuple[str, int], timeout: float = 1):
        self.peer_sock = peer_sock
        self.peer_host = peer_addr[0]
        self.peer_port = peer_addr[1]
        self.chunk_size = RECV_CHUNK_SIZE
        self.peer_sock.settimeout(timeout)

    def _fetch_bytes(self, num_bytes: int) -> bytes:
        """Receive exactly num_bytes or raise RuntimeError on closed connection."""
        collected = []
        total_received = 0
        while total_received < num_bytes:
            fragment = self.peer_sock.recv(min(self.chunk_size, num_bytes - total_received))
            if fragment == b"":
                raise RuntimeError("Connection closed while receiving data")
            collected.append(fragment)
            total_received += len(fragment)
        return b"".join(collected)

    def send(self, msg: str) -> int:
        """Send a UTF-8 message with 8-byte length prefix.

        Returns:
            Number of payload bytes sent (excluding header).
        """
        if not isinstance(msg, str):
            _LOGGER.warning("Failed to send data. Expected str, got %s", type(msg))
            return -1

        encoded = msg.encode("utf-8")
        msg_len = len(encoded)
        frame_header = str(msg_len).zfill(PREFIX_BYTES).encode("utf-8")

        self.peer_sock.sendall(frame_header)

        if msg_len > 0:
            self.peer_sock.sendall(encoded)

        return msg_len

    def read(self) -> str:
        """Read one framed UTF-8 message."""
        frame_header = self._fetch_bytes(PREFIX_BYTES)
        try:
            msg_len = int(frame_header.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError("Invalid message header received") from exc

        if msg_len == 0:
            return ""

        encoded = self._fetch_bytes(msg_len)
        return encoded.decode("utf-8")


# Backward compatibility alias
Connection = PeerConnection


class Receiver:
    """TCP receiver/connector that manages one or more `PeerConnection` peers."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = SocketWrapper(host, port)
        self.socket.set_timeout(1.0)

        self.listening_thread = DaemonThread(target=self.start_listening, args=())
        self.peers: List[PeerConnection] = []
        self.already_warned = False

    def start_listening(self):
        """Accept connections in a loop and handshake each peer."""
        if not self.peers:
            self.socket.bind()
            self.socket.listen()

        _LOGGER.info("Starting TCP server listening @tcp://%s:%s", self.host, self.port)

        while not self.listening_thread.should_halt():
            try:
                incoming_sock, incoming_addr = self.socket.accept_connections()
                _LOGGER.info("Incoming connection from %s:%s", incoming_addr[0], incoming_addr[1])

                peer = PeerConnection(incoming_sock, incoming_addr)
                if self.perform_sync(peer):
                    self.peers.append(peer)
                    _LOGGER.info("Connection added: %s:%s", peer.peer_host, peer.peer_port)
                    self.clear_buffer()
                else:
                    _LOGGER.warning("Handshake failed for %s:%s", peer.peer_host, peer.peer_port)

            except socket.timeout:
                _LOGGER.debug("Socket timed out waiting for new connections")
            except RuntimeError as err:
                _LOGGER.warning("Connection setup failed: %s", err)

        _LOGGER.info("Server halted")
        self.peers = []

    def initiate_connection(self, host: str, port: int, timeout: float = 3):
        """Create an outbound TCP connection and perform handshake."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (OSError, AttributeError):
            _LOGGER.debug("SO_REUSEPORT is not available on this platform")

        sock.settimeout(timeout)
        sock.connect((host, port))

        peer = PeerConnection(sock, (host, port), timeout=timeout)
        _LOGGER.info("Initiating handshake @tcp://%s:%s", host, port)

        try:
            if self.perform_sync(peer):
                self.peers.append(peer)
                _LOGGER.info("Connection added: %s:%s", peer.peer_host, peer.peer_port)
                print(f"Connection to {host}:{port} added.")
            else:
                _LOGGER.warning("Handshake failed for %s:%s", host, port)

        except socket.timeout:
            _LOGGER.debug("Socket timed out while waiting for handshake response")
        except RuntimeError as err:
            _LOGGER.warning("Handshake failed: %s", err)

    def clear_buffer(self):
        """Drain first non-empty buffered message after connection setup."""
        attempts = 0
        msg = ""
        while msg == "":
            msg = self.conn_read()
            attempts += 1
        _LOGGER.info("Buffer cleared after %s reads", attempts)

    def conn_send(self, msg: str) -> int:
        """Broadcast a message to all active connections.

        Returns:
            Sum of payload bytes sent successfully.
        """
        total = 0
        for peer in self.peers:
            sent = peer.send(msg)
            if sent == -1:
                _LOGGER.warning("Failed to send data to %s:%s", peer.peer_host, peer.peer_port)
                continue
            total += sent
        return total

    def conn_read(self) -> str:
        """Read and concatenate one message from each active connection."""
        results = []
        for peer in self.peers:
            try:
                results.append(peer.read())
                self.already_warned = False
            except socket.timeout:
                if not self.already_warned:
                    _LOGGER.warning("Waiting for a response...")
                self.already_warned = True
                results.append("")
        return "".join(results)

    @staticmethod
    def perform_sync(peer: PeerConnection) -> bool:
        """Perform symmetric sync-marker handshake with a peer."""
        peer.send(SYNC_MARKER)
        _LOGGER.debug("Sent handshake token")
        response = peer.read()
        _LOGGER.debug("Handshake response: %s", response)
        return response == SYNC_MARKER

    def start_server(self):
        """Start background listening thread."""
        self.listening_thread.start()


class DaemonThread(threading.Thread):
    """Thread wrapper with cooperative halt event."""

    def __init__(self, target, args):
        super(DaemonThread, self).__init__(target=target, args=args)
        self.stop_flag = threading.Event()

    def should_halt(self):
        """Return True if halt event is set."""
        return self.stop_flag.is_set()

    def request_stop(self):
        """Set halt event."""
        self.stop_flag.set()

    # Backward compatibility methods
    def halt_isSet(self):
        """Return True if halt event is set (deprecated)."""
        return self.should_halt()

    def halt_set(self):
        """Set halt event (deprecated)."""
        self.request_stop()


# Backward compatibility aliases
HaltingThread = DaemonThread
Listener = Receiver
