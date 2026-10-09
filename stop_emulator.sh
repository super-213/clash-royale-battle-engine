#!/bin/zsh
set -eu

ANDROID_SDK_DIR=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-$HOME/Library/Android/sdk}}
ADB_BIN=$ANDROID_SDK_DIR/platform-tools/adb
EMULATOR_SERIAL=${CLASH_TRAIN_SERIAL:-emulator-5554}

if [[ ! -x "$ADB_BIN" ]]; then
  print -u2 "找不到 adb：$ADB_BIN"
  exit 1
fi
if [[ "$EMULATOR_SERIAL" != emulator-* ]]; then
  print -u2 "只允许关闭 Android 模拟器：$EMULATOR_SERIAL"
  exit 1
fi
DEVICE_STATE=$("$ADB_BIN" devices | awk -v serial="$EMULATOR_SERIAL" '$1 == serial {print $2}')
if [[ -z "$DEVICE_STATE" ]]; then
  print "[emulator] $EMULATOR_SERIAL 未运行"
  exit 0
fi
if [[ "$DEVICE_STATE" == device ]]; then
  "$ADB_BIN" -s "$EMULATOR_SERIAL" emu kill
fi

for (( attempt = 0; attempt < 30; attempt++ )); do
  DEVICE_STATE=$("$ADB_BIN" devices | awk -v serial="$EMULATOR_SERIAL" '$1 == serial {print $2}')
  if [[ -z "$DEVICE_STATE" ]]; then
    print "[emulator] $EMULATOR_SERIAL 已关闭"
    exit 0
  fi
  sleep 1
done
print -u2 "关闭模拟器超时：$EMULATOR_SERIAL"
exit 1
