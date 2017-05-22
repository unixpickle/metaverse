#!/bin/bash

if [ ! -f flashgames.json ]; then
  RAW_URL='https://raw.githubusercontent.com/openai/universe/master/universe/runtimes/flashgames.json'
  curl "$RAW_URL" >flashgames.json || exit 1
fi

cat flashgames.json |
  jq 'with_entries(select(.value.rewarder == true))' |
  jq 'with_entries(select(.value.autostart == true))'
