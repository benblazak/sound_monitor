import json
import logging

_logger = logging.getLogger(__name__)

try:
    from ben.mail import mail as _mail

    def mail(*, subject: str, body: str) -> None:
        _mail(subject=subject, body=body)

except ImportError:

    def mail(*, subject: str, body: str) -> None:
        _logger.info(
            "attempted to send mail\n"
            + json.dumps(
                {
                    "subject": subject,
                    "body": body[:100] + ("..." if len(body) > 100 else ""),
                },
                indent=2,
            ),
        )
