

## Example inference

``` console
curl -X POST \
    --header 'Content-Type: application/json' \
    --data "@data/iris-oip.json" \
    http://localhost:8080/v2/models/basic-iris/versions/v0.0.1/infer
```



## References

### Open Inference Protocol
- [doc](https://github.com/kserve/open-inference-protocol/blob/main/specification/protocol/inference_rest.md)