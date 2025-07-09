"""Entity for conversation integration."""

from abc import abstractmethod
import logging
from typing import Literal, final

from homeassistant.const import STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.chat_session import async_get_chat_session
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .chat_log import AssistantContent, ChatLog, ToolResultContent, async_get_chat_log
from .const import ConversationEntityFeature
from .models import ConversationInput, ConversationResult


class ConversationEntity(RestoreEntity):
    """Entity that supports conversations."""

    _attr_should_poll = False
    _attr_supported_features = ConversationEntityFeature(0)
    _attr_supports_streaming = False
    __last_activity: str | None = None

    @property
    def supports_streaming(self) -> bool:
        """Return if the entity supports streaming responses."""
        return self._attr_supports_streaming

    @property
    @final
    def state(self) -> str | None:
        """Return the state of the entity."""
        if self.__last_activity is None:
            return None
        return self.__last_activity

    async def async_internal_added_to_hass(self) -> None:
        """Call when the entity is added to hass."""
        await super().async_internal_added_to_hass()
        state = await self.async_get_last_state()
        if (
            state is not None
            and state.state is not None
            and state.state not in (STATE_UNAVAILABLE, STATE_UNKNOWN)
        ):
            self.__last_activity = state.state

    @final
    async def internal_async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""
        self.__last_activity = dt_util.utcnow().isoformat()
        self.async_write_ha_state()
        return await self.async_process(user_input)

    @property
    @abstractmethod
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a sentence."""
        with (
            async_get_chat_session(self.hass, user_input.conversation_id) as session,
            async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Call the API."""
        raise NotImplementedError

    async def async_prepare(self, language: str | None = None) -> None:
        """Load intents for a language."""

    @callback
    def _async_get_result_from_chat_log(
        self, user_input: ConversationInput, chat_log: ChatLog
    ) -> ConversationResult:
        """Get the result from the chat log."""
        tool_results = [
            content.tool_result
            for content in chat_log.content[chat_log.llm_input_provided_index :]
            if isinstance(content, ToolResultContent)
            and isinstance(content.tool_result, llm.IntentResponseDict)
        ]

        if tool_results:
            intent_response = tool_results[-1].original
        else:
            intent_response = intent.IntentResponse(language=user_input.language)

        if not isinstance((last_content := chat_log.content[-1]), AssistantContent):
            logging.getLogger(__name__).error(
                "Last content in %s chat log is not an AssistantContent: %s. This could be due to the model not returning a valid response",
                self.entity_id,
                last_content,
            )
            raise HomeAssistantError("Unable to get response")

        intent_response.async_set_speech(last_content.content or "")

        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )
