# MLExchange



## How to Run the Demo

First, make sure you have docker-compose working, then run:
```
docker-compose build
docker-compose up
```

You have to manually start the machine learning server (just in case it breaks, so you can restart it).

In another terminal:
```
docker exec -it cons /bin/bash
```

This punts you into the ml server, and then you can start the messaging queue listener, so you can listen
for ml tasks that the web ui has dispatched.

Now run (this starts the messaging queue listener):
```
python3 app/ml_working.py
```

Now, you should be good to go!



