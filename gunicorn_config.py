import os

workers = int(os.environ.get('GUNICORN_PROCESSES', '2'))
threads = int(os.environ.get('GUNICORN_THREADS', '4'))
# timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))
bind = os.environ.get('GUNICORN_BIND', '127.0.0.1:5000')

# Note: OLCF SSL gateway will forward
# the following headers on secure requests
# https://docs.olcf.ornl.gov/services_and_applications/slate/networking/route.html#optional-application-authentication
#
# Host: nginx-echo-headers-stf002platform.bedrock-dev.ccs.ornl.gov
# X-Remote-User: kincljc
# X-Forwarded-Host: nginx-echo-headers-stf002platform.bedrock-dev.ccs.ornl.gov
# X-Forwarded-Port: 443
# X-Forwarded-Proto: https
# Forwarded: for=160.91.195.36;host=nginx-echo-headers-stf002platform.bedrock-dev.ccs.ornl.gov;proto=https;proto-version=
# X-Forwarded-For: 160.91.195.36

# forwarded_allow_ips = "*"
secure_scheme_headers = {
    "X-Forwarded-Proto": "https"
}
