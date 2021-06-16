# Text2Pose

This module generates pose vector from given text.

## Launching

- **build & run module**

    ```
    docker build -t model .
    docker run -d -p 1488:1488 --rm  --name model_running model
    ```

    reading logs:
    ```
    docker logs -f model_running
    ```

    stopping:
    ```
    docker stop model_running
    ```


## Usage

```
curl -X POST -d "text=girl is sitting on a chair" "http://localhost:1488/get_pose/"
```

answer:
```
{
   "body":[
      [
         0.1252046525478363,
         0.28800007700920105
      ],
      [
         0.10787834227085114,
         0.0004583969712257385
      ],
      ...
   ]
}
```
