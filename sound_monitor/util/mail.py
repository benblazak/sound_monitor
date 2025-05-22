import json
import logging
from typing import Sequence

_logger = logging.getLogger(__name__)

try:
    from ben.mail import mail as _mail

    available = True

    def mail(
        *,
        subject: str,
        body: str,
        attachments: str | Sequence[str] | None = None,
    ) -> None:
        _mail(
            subject=subject,
            body=body,
            attachments=attachments,
        )

except ImportError:

    available = False

    def mail(
        *,
        subject: str,
        body: str,
        attachments: str | Sequence[str] | None = None,
    ) -> None:

        if attachments is None:
            attachments = []
        if isinstance(attachments, str):
            attachments = [attachments]
        attachments = [str(e) for e in attachments]

        _logger.info(
            "attempted to send mail\n"
            + json.dumps(
                {
                    "subject": subject,
                    "attachments": attachments,
                    "body": body[:100] + ("..." if len(body) > 100 else ""),
                },
                indent=2,
            ),
        )
