import asyncio
import logging
from enum import Enum, auto
from typing import Any, List, Optional

import websockets.client as ws


class PopupError(Exception):
    def __init__(self, *args: Any):
        super().__init__(*args)


class MessageType(Enum):
    LOGIN = auto()
    GAMES = auto()
    CHALLENGE = auto()
    CANCEL = auto()
    ACCEPT = auto()
    OBSERVE = auto()
    LEAVE = auto()


class Client:
    username: str
    logger: logging.Logger
    room: Optional[str]
    websocket: Optional[ws.WebSocketClientProtocol]

    def __init__(self, username: str):
        self.username = username
        self.logger = logging.getLogger(username)
        self.room = None
        self.websocket = None

    async def connect(self):
        while True:
            try:
                self.websocket = await ws.connect(
                    "ws://localhost:8000/showdown/websocket"
                )
                break
            except (ConnectionRefusedError, TimeoutError):
                self.logger.error("Connection attempt failed, retrying now")
                await asyncio.sleep(10)

    async def send_message(self, message: str):
        room_str = self.room or ""
        message = f"{room_str}|{message}"
        self.logger.info(message)
        if self.websocket:
            await self.websocket.send(message)
        else:
            raise ConnectionError("Cannot send message without established websocket")

    async def receive_message(self) -> str:
        if self.websocket:
            response = str(await self.websocket.recv())
        else:
            raise ConnectionError(
                "Cannot receive message without established websocket"
            )
        self.logger.info(response)
        return response

    async def find_message(self, message_type: MessageType) -> List[str]:
        while True:
            message = await self.receive_message()
            split_message = message.split("|")
            if message_type == MessageType.LOGIN:
                if split_message[1] == "challstr":
                    return split_message
            elif message_type == MessageType.GAMES:
                if split_message[1] == "popup":
                    # Popups encountered when searching for games message in the past:
                    # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                    # NOTE: This popup occurs in response to the player accepting a challenge, but manifests when looking for
                    # the games message.
                    raise PopupError(split_message[2])
                elif split_message[1] == "updatesearch":
                    return split_message
            elif message_type == MessageType.CHALLENGE:
                if split_message[1] == "popup":
                    # Popups encountered when searching for challenge message in the past:
                    # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                    # 2. You challenged less than 10 seconds after your last challenge! It's cancelled in case it's a misclick.
                    # 3. You are already challenging someone. Cancel that challenge before challenging someone else.
                    # 4. The server is restarting. Battles will be available again in a few minutes.
                    raise PopupError(split_message[2])
                elif split_message[1] == "pm":
                    if (
                        split_message[2] == f"!{self.username}"
                        and "Due to spam from your internet provider, you can't challenge others right now."
                        in split_message[4]
                    ):
                        raise PopupError(split_message[4])
                    elif (
                        split_message[2] == f" {self.username}"
                        and "wants to battle!" in split_message[4]
                    ):
                        return split_message
            elif message_type == MessageType.CANCEL:
                if split_message[1] == "popup":
                    # Popups encountered when searching for cancel message in the past:
                    # 1. You are not challenging <opponent_username>. Maybe they accepted/rejected before you cancelled?
                    raise PopupError(split_message[2])
                elif (
                    split_message[1] == "pm"
                    and split_message[2] == f" {self.username}"
                    and "cancelled the challenge." in split_message[4]
                ):
                    return split_message
            elif message_type == MessageType.ACCEPT:
                if (
                    split_message[1] == "pm"
                    and split_message[3] == f" {self.username}"
                    and "wants to battle!" in split_message[4]
                ):
                    return split_message
            elif message_type == MessageType.OBSERVE:
                is_request = split_message[1] == "request" and split_message[2]
                is_protocol = "\n" in split_message
                if is_request or is_protocol:
                    return split_message
            elif message_type == MessageType.LEAVE:
                if (
                    self.room is not None
                    and self.room in split_message[0]
                    and split_message[1] == "deinit"
                ):
                    return split_message
