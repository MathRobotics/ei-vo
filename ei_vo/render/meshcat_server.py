"""MeshCat server wrapper with configurable port ranges."""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import sys
import webbrowser

from meshcat.servers import zmqserver as base


def _read_port_range(name: str, *, default_start: int, default_end: int) -> tuple[int, int]:
    start = int(os.environ.get(f"EI_VO_MESHCAT_{name}_PORT_START", default_start))
    end = int(os.environ.get(f"EI_VO_MESHCAT_{name}_PORT_END", default_end))
    if start <= 0:
        raise ValueError(f"{name} port start must be positive. Got {start}.")
    if end <= start:
        raise ValueError(f"{name} port end must be greater than start. Got start={start}, end={end}.")
    return start, end


class EIVOMeshcatBridge(base.ZMQWebSocketBridge):
    """Upstream bridge with configurable ZMQ and web port scan ranges."""

    def __init__(
        self,
        zmq_url=None,
        host="127.0.0.1",
        port=None,
        certfile=None,
        keyfile=None,
        ngrok_http_tunnel=False,
    ):
        self.host = host
        self.websocket_pool = set()
        self.app = self.make_app()
        self.ioloop = base.tornado.ioloop.IOLoop.current()

        default_zmq_end = base.DEFAULT_ZMQ_PORT + base.MAX_ATTEMPTS
        zmq_port_start, zmq_port_end = _read_port_range(
            "ZMQ",
            default_start=base.DEFAULT_ZMQ_PORT,
            default_end=default_zmq_end,
        )
        default_web_end = base.DEFAULT_FILESERVER_PORT + base.MAX_ATTEMPTS
        web_port_start, web_port_end = _read_port_range(
            "WEB",
            default_start=base.DEFAULT_FILESERVER_PORT,
            default_end=default_web_end,
        )

        if zmq_url is None:
            def setup_zmq_at_port(candidate_port: int):
                url = f"{base.DEFAULT_ZMQ_METHOD}://{self.host}:{candidate_port}"
                return self.setup_zmq(url)

            (self.zmq_socket, self.zmq_stream, self.zmq_url), _ = base.find_available_port(
                setup_zmq_at_port,
                zmq_port_start,
                max_attempts=zmq_port_end - zmq_port_start,
            )
        else:
            self.zmq_socket, self.zmq_stream, self.zmq_url = self.setup_zmq(zmq_url)

        protocol = "http:"
        listen_kwargs = {}
        if certfile is not None or keyfile is not None:
            if certfile is None:
                raise Exception("You must supply a certfile if you supply a keyfile")
            if keyfile is None:
                raise Exception("You must supply a keyfile if you supply a certfile")
            listen_kwargs["ssl_options"] = {"certfile": certfile, "keyfile": keyfile}
            protocol = "https:"

        if port is None:
            _, self.fileserver_port = base.find_available_port(
                self.app.listen,
                web_port_start,
                max_attempts=web_port_end - web_port_start,
                **listen_kwargs,
            )
        else:
            self.app.listen(port, **listen_kwargs)
            self.fileserver_port = port

        self.web_url = "{protocol}//{host}:{port}/static/".format(
            protocol=protocol,
            host=self.host,
            port=self.fileserver_port,
        )

        if ngrok_http_tunnel:
            if protocol == "https:":
                raise Exception("The free version of ngrok does not support https")
            try:
                import pyngrok.conf
                import pyngrok.ngrok
            except ImportError as exc:
                raise RuntimeError("pyngrok is required when ngrok_http_tunnel is enabled.") from exc
            config = pyngrok.conf.PyngrokConfig(start_new_session=True)
            self.web_url = pyngrok.ngrok.connect(self.fileserver_port, "http", pyngrok_config=config)


def main() -> None:
    if sys.version_info >= (3, 8) and platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="Serve MeshCat with configurable port ranges.")
    parser.add_argument("--zmq-url", "-z", type=str, nargs="?", default=None)
    parser.add_argument("--open", "-o", action="store_true")
    parser.add_argument("--certfile", type=str, default=None)
    parser.add_argument("--keyfile", type=str, default=None)
    parser.add_argument("--ngrok_http_tunnel", action="store_true")
    results = parser.parse_args()

    bridge = EIVOMeshcatBridge(
        zmq_url=results.zmq_url,
        certfile=results.certfile,
        keyfile=results.keyfile,
        ngrok_http_tunnel=results.ngrok_http_tunnel,
    )
    print(f"zmq_url={bridge.zmq_url}")
    print(f"web_url={bridge.web_url}")
    if results.open:
        webbrowser.open(bridge.web_url, new=2)

    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
