import matlab.engine
# from mlab.releases import latest_release
# from matlab import matlabroot
# print(matlabroot())
eng = matlab.engine.start_matlab()
eng.cd('/home/hukcc/HYY-WORKSPACE/funcs/',nargout=0)
# y_raw,Fs = eng.audioread("/home/hukcc/HYY-WORKSPACE/funcs/tongguo.wav", matlab.double([[1,0]]), nargout=2)
# Q1,Q2,Q3 = eng.AudioDivisionFunc(y_raw,Fs,nargout=3)
Q1,Q2,Q3 = eng.test("/home/hukcc/HYY-WORKSPACE/funcs/tongguo110.wav",nargout=3)

# ret = eng.CoderFuncTest(nargout=0)

print("test")