#!/bin/bash

MAIN_BRANCH="main"

# 读取版本号
VERSION=$(cat version.txt)

# 设置IFS来分隔版本号
OLD_IFS="$IFS"
IFS='.' read -ra strArr <<< "$VERSION"

# 计算新的尾部版本号
TAILVER=${strArr[2]}
NEW_TAILVER=$((TAILVER + 1))

# 拼接新的版本号
NEW_VER="${strArr[0]}.${strArr[1]}.$NEW_TAILVER"

# 恢复原始的IFS
IFS="$OLD_IFS"

VER=$NEW_VER \
    && echo $VER > version.txt \
    && git checkout $MAIN_BRANCH \
    && git add . \
    && git commit -m "update version $VER" \
    && git tag $VER \
    && git push origin $MAIN_BRANCH --tags