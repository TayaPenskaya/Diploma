# Segmentation

This module generates detailed segmentation.

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
curl -X POST -d "image=" "http://localhost:1488/get_segmentation/"
```

answer:
```
{
   
}
```

