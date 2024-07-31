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
    def __init__(self, username: str):
        self.username = username
        self.logger = logging.getLogger(username)
        self.room = None
        self.websocket = None
        self.loop = asyncio.get_event_loop()

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

    async def receive_message(self, timeout: Optional[float] = None) -> str:
        if self.websocket is not None:
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout)
                self.logger.info(response)
                return str(response)
            except asyncio.TimeoutError:
                raise TimeoutError("Timed out waiting for a message")
        else:
            raise ConnectionError(
                "Cannot receive message without established websocket"
            )

    async def find_message(
        self, message_type: MessageType, timeout: Optional[float] = None
    ) -> List[List[str]]:
        while True:
            message = await self.receive_message(timeout)
            split_messages = [m.split("|") for m in message.split("\n")]
            if message_type == MessageType.LOGIN:
                if split_messages[0][1] == "challstr":
                    return split_messages
            elif message_type == MessageType.GAMES:
                if split_messages[0][1] == "popup":
                    # Popups encountered when searching for games message in the past:
                    # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                    # NOTE: This popup occurs in response to the player accepting a challenge, but manifests when looking for
                    # the games message.
                    raise PopupError(split_messages[0][2])
                elif split_messages[0][1] == "updatesearch":
                    return split_messages
            elif message_type == MessageType.CHALLENGE:
                if len(split_messages[0]) > 1 and split_messages[0][1] == "popup":
                    # Popups encountered when searching for challenge message in the past:
                    # 1. Due to high load, you are limited to 12 battles and team validations every 3 minutes.
                    # 2. You challenged less than 10 seconds after your last challenge! It's cancelled in case it's a misclick.
                    # 3. You are already challenging someone. Cancel that challenge before challenging someone else.
                    # 4. The server is restarting. Battles will be available again in a few minutes.
                    raise PopupError(split_messages[0][2])
                elif len(split_messages[0]) > 1 and split_messages[0][1] == "pm":
                    if (
                        split_messages[0][2] == f"!{self.username}"
                        and "Due to spam from your internet provider, you can't challenge others right now."
                        in split_messages[0][4]
                    ):
                        raise PopupError(split_messages[0][4])
                    elif (
                        split_messages[0][2] == f" {self.username}"
                        and "wants to battle!" in split_messages[0][4]
                    ):
                        return split_messages
            elif message_type == MessageType.CANCEL:
                if split_messages[0][1] == "popup":
                    # Popups encountered when searching for cancel message in the past:
                    # 1. You are not challenging <opponent_username>. Maybe they accepted/rejected before you cancelled?
                    raise PopupError(split_messages[0][2])
                elif (
                    split_messages[0][1] == "pm"
                    and split_messages[0][2] == f" {self.username}"
                    and "cancelled the challenge." in split_messages[0][4]
                ):
                    return split_messages
            elif message_type == MessageType.ACCEPT:
                if (
                    split_messages[0][1] == "pm"
                    and split_messages[0][3] == f" {self.username}"
                    and "wants to battle!" in split_messages[0][4]
                ):
                    return split_messages
            elif message_type == MessageType.OBSERVE:
                is_request = (
                    len(split_messages) == 2
                    and len(split_messages[1]) == 3
                    and split_messages[1][1] == "request"
                    and split_messages[1][2]
                )
                is_protocol = ["", ""] in split_messages
                if is_request or is_protocol:
                    return split_messages
            elif message_type == MessageType.LEAVE:
                if (
                    self.room is not None
                    and self.room in split_messages[0][0]
                    and len(split_messages[1]) > 1
                    and split_messages[1][1] == "deinit"
                ):
                    return split_messages
