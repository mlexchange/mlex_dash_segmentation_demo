#!/bin/bash

while true
do
    /usr/bin/amqp-consume --url=$AMQP_URL -q ml_tasks -c 1 cat && echo
done

