"""Create and load E-mail block (Prefect)."""
import os
import typing

from prefect_email import EmailServerCredentials


EMAIL_ADDRESS_VAR: typing.Final[str] = "GMAIL_ADDRESS"
EMAIL_APP_PASSWORD_VAR: typing.Final[str] = "GMAIL_APP_PASSWORD"

EMAIL_CREDS_NAME: typing.Final[str] = "gmail-creds"


def create_email_creds_block():
    credentials = EmailServerCredentials(
        username=os.getenv(EMAIL_ADDRESS_VAR, default="test@mail.com"),
        password=os.getenv(EMAIL_APP_PASSWORD_VAR, default=""),
    )
    credentials.save(name=EMAIL_CREDS_NAME, overwrite=True)


if __name__ == "__main__":
    create_email_creds_block()
