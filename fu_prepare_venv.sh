#!/usr/bin/bash
cp ~/opt/local-v1.13-langmo-mod.tgz /local/
cd /local
pigz -dc local-v1.13-langmo-mod.tgz | tar xf -

echo "setting up venv done"
