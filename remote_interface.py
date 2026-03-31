"""Length-prefixed TCP messaging layer for distributed communication.

Provides low-level TCP socket abstractions and framed-message protocol for
reliable inter-process communication. Messages use 8-byte ASCII length header
followed by UTF-8 payload.

Key Features:
    - Length-prefixed framing to ensure message boundaries
    - Symmetric handshake protocol with sync markers
    - Thread-safe peer connection management
    - Cooperative thread termination via halt events
    - Backward compatibility aliases for deprecated class names

Typical Usage:
    Server side: receiver = Receiver('127.0.0.1', 9999)
                  receiver.start_server()
    Client side: receiver.initiate_connection('127.0.0.1', 9999)
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
    """TCP socket wrapper with reusable defaults and connection utilities.
    
    Abstracts socket creation, binding, and connection with sensible defaults
    for timeout and socket options. Handles platform-specific socket options
    (e.g., SO_REUSEPORT may not be available on all platforms).
    
    Args:
        host: IP address or hostname to bind/connect to.
        service_port: Port number for binding/connecting.
    """

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
        """Configure socket timeout in seconds.
        
        Args:
            timeout_sec: Timeout duration in seconds. Use 0 for non-blocking mode.
        """
        self.sock.settimeout(timeout_sec)

    def listen(self):
        """Put socket into listening mode.
        
        Called after bind() to start accepting connections.
        """
        self.sock.listen()

    def bind(self):
        """Bind socket to configured address and port.
        
        Raises:
            ValueError: If host or service_port not set during initialization.
        """
        if self.host is None or self.service_port is None:
            _LOGGER.error(
                "Socket configured without address/port: (%s:%s)",
                self.host,
                self.service_port,
            )
            raise ValueError("Socket address and port must be set before bind()")
        self.sock.bind((self.host, self.service_port))

    def accept_connections(self):
        """Accept one incoming connection.
        
        Returns:
            Tuple of (socket, (host, port)) for new connection.
        """
        return self.sock.accept()

    def connect(self):
        """Connect socket to configured address and port.
        
        Raises:
            ValueError: If host or service_port not set during initialization.
            socket.error: If connection fails.
        """
        if self.host is None or self.service_port is None:
            raise ValueError("Socket address and port must be set before connect()")
        _LOGGER.info("Connecting to (%s,%s)", self.host, self.service_port)
        self.sock.connect((self.host, self.service_port))

    def read(self):
        """Read up to RECV_CHUNK_SIZE bytes and decode as UTF-8.
        
        Returns:
            Decoded string, or empty string on zero-byte read.
        """
        return self.sock.recv(RECV_CHUNK_SIZE).decode("utf-8")

    def send(self, data):
        """Send raw bytes payload.
        
        Args:
            data: Bytes to transmit.
        """
        self.sock.send(data)


# Backward compatibility alias
TCPSocket = SocketWrapper


class PeerConnection:
    """Single framed-message TCP connection with length-prefixed protocol.
    
    Handles reliable message transmission/reception using 8-byte length prefix
    (ASCII, zero-padded) followed by UTF-8 payload. Ensures complete message
    delivery by chunking reads/writes as needed.
    
    Args:
        peer_sock: Connected socket object.
        peer_addr: Tuple of (host, port) for logging/identification.
        timeout: Read timeout in seconds.
    """

    def __init__(self, peer_sock: socket.socket, peer_addr: Tuple[str, int], timeout: float = 1):
        self.peer_sock = peer_sock
        self.peer_host = peer_addr[0]
        self.peer_port = peer_addr[1]
        self.chunk_size = RECV_CHUNK_SIZE
        self.peer_sock.settimeout(timeout)

    def _fetch_bytes(self, num_bytes: int) -> bytes:
        """Receive exactly num_bytes from peer or raise on closed connection.
        
        Args:
            num_bytes: Exact number of bytes to receive.
            
        Returns:
            Bytes received (guaranteed to be num_bytes length).
            
        Raises:
            RuntimeError: If connection closed before num_bytes received.
        """
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
        """Send message with 8-byte length prefix.
        
        Args:
            msg: Message string to send.
            
        Returns:
            Number of payload bytes sent (excluding header), or -1 on type error.
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
        """Read one length-prefixed message from peer.
        
        Returns:
            Decoded message string, or empty string for zero-length message.
            
        Raises:
            RuntimeError: On invalid header or closed connection.
        """
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
    """Manages incoming and outgoing TCP peer connections with framing.
    
    Provides both server-side (listening for connections) and client-side
    (initiating connections) functionality. Automatically performs handshake
    with symmetric sync markers for peer verification. All peer connections
    are stored and managed for broadcast/aggregated communication.
    
    Args:
        host: IP address to listen on or connect to.
        port: Port number for listener/client.
    
    Attributes:
        peers: List of active PeerConnection objects.
        listening_thread: Background thread managing connections.
    """

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = SocketWrapper(host, port)
        self.socket.set_timeout(1.0)

        self.listening_thread = DaemonThread(target=self.start_listening, args=())
        self.peers: List[PeerConnection] = []
        self.already_warned = False

    def start_listening(self):
        """Accept connections in background loop, handshaking each peer.
        
        Binds socket to host:port, listens for connections, performs
        handshake with each peer, and manages connection lifecycle.
        Runs until halt event is signaled.
        """
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
        """Initiate outbound TCP connection and perform handshake.
        
        Creates client socket, connects to specified host:port, and performs
        synchronization handshake. Connection is added to peers list on success.
        
        Args:
            host: Target IP address or hostname.
            port: Target port number.
            timeout: Connection/handshake timeout in seconds.
        """
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
        """Drain first non-empty message from all peers after setup.
        
        Used to clear handshake artifacts or buffered initial messages.
        Loops until at least one peer returns non-empty message.
        """
        attempts = 0
        msg = ""
        while msg == "":
            msg = self.conn_read()
            attempts += 1
        _LOGGER.info("Buffer cleared after %s reads", attempts)

    def conn_send(self, msg: str) -> int:
        """Broadcast message to all active peer connections.
        
        Args:
            msg: Message string to broadcast.
            
        Returns:
            Total payload bytes sent across all connections.
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
        """Read one message from each peer and concatenate results.
        
        Returns:
            Concatenated messages from all peers. Uses empty string for
            peers with timeout/no data.
        """
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
        """Perform symmetric handshake with SYNC_MARKER.
        
        Args:
            peer: PeerConnection to handshake with.
            
        Returns:
            True if peer responds with matching sync marker, False otherwise.
        """
        peer.send(SYNC_MARKER)
        _LOGGER.debug("Sent handshake token")
        response = peer.read()
        _LOGGER.debug("Handshake response: %s", response)
        return response == SYNC_MARKER

    def start_server(self):
        """Start background listening thread for accepting connections.
        
        Spawns DaemonThread that runs start_listening() in background.
        """
        self.listening_thread.start()


class DaemonThread(threading.Thread):
    """Thread wrapper with cooperative halt signaling via event.
    
    Provides graceful thread termination using threading.Event rather than
    forceful termination. Target function should periodically check should_halt().
    
    Args:
        target: Callable to run in thread.
        args: Arguments to pass to target.
    """

    def __init__(self, target, args):
        super(DaemonThread, self).__init__(target=target, args=args)
        self.stop_flag = threading.Event()

    def should_halt(self):
        """Check if halt event is set.
        
        Returns:
            True if halt requested, False otherwise.
        """
        return self.stop_flag.is_set()

    def request_stop(self):
        """Request thread graceful stop via halt event.
        
        Signals that thread should terminate. Target function must
        periodically call should_halt() to check this flag.
        """
        self.stop_flag.set()

    # Backward compatibility aliases
    def halt_isSet(self):
        """Check halt event (deprecated).
        
        Deprecated:
            Use should_halt() instead for clearer intent.
            
        Returns:
            True if halt event is set.
        """
        return self.should_halt()

    def halt_set(self):
        """Request stop (deprecated).
        
        Deprecated:
            Use request_stop() instead for clearer intent.
        """
        self.request_stop()
