#!/bin/bash

CERTS=/sdf/home/r/rogersdd/venvs/dsn/etc/certs

# the --ssl-cert-reqs 2 option requires the client to
# authenticate with a certificate (mutual TLS)
#
# Note: --ssl-cert-reqs 1 is insufficient for mTLS!
# This is a bug in uvicorn's documentation!
#
#   >>> import ssl
#   >>> ssl.CERT_REQUIRED
#   <VerifyMode.CERT_REQUIRED: 2>
#   >>> ssl.CERT_OPTIONAL
#   <VerifyMode.CERT_OPTIONAL: 1>
#
#
# the client certificate must have been signed
# with the root specified in --ssl-ca-certs.
uvicorn --ssl-keyfile $CERTS/sdfdtn.key \
	--ssl-certfile $CERTS/sdfdtn.pem \
        --ssl-cert-reqs 2 \
	--ssl-ca-certs $CERTS/ca_root.pem \
	--host 0.0.0.0 \
	--port 8000 \
        lclstream.server:app

