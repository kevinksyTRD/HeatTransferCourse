# Installation of CoolProp
## Trouble shooting
if error below occurs,
```commandline
ImportError: dlopen(/venv/lib/python3.8/site-packages/CoolProp/_constants.cpython-38-darwin.so, 2): Library not loaded: @rpath/libc++.1.dylib
  Referenced from: /venv/lib/python3.8/site-packages/CoolProp/_constants.cpython-38-darwin.so
  Reason: image not found
```
Find the location of "libc++.1.dylib". If not found, consider installing it from 
[this website](https://libcxx.llvm.org). 

After the file is found or installed, 
```commandline
install_name_tool -change "@rpath/libc++.1.dylib" "/Users/keviny/Dev/llvm-project/build/lib/libc++.1.dylib" /Users/keviny/Documents/teaching/TMR4222-Heat-Transfer/HeatTransferCourse/venv/lib/python3.8/site-packages/CoolProp/_constants.cpython-38-darwin.so 
```
