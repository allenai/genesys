NUM=1

CURRENT=${NUM}
IMAGE_NAME=model_discovery
DOCKERFILE_NAME=Docker.beaker

GIT_HASH=`git log --format="%h" -n 1`
IMAGE=$IMAGE_NAME_$USER-$GIT_HASH
IM_NAME=${IMAGE}_${NUM}

echo "Building $IMAGE"
echo ${GITHUB_TOKEN}
docker build --build-arg GITHUB=${GITHUB_TOKEN} --platform linux/amd64 --load -f $DOCKERFILE_NAME -t $IMAGE .


echo "Now uploading to beaker"
echo ${IMAGE}
beaker image create --name=${IM_NAME}_cuda111_${CURRENT} --description="modeldiscovery${CURRENT}_${GIT_HASH}" $IMAGE
