import cuorennAudioNN as nnAudio
import time


start = time.perf_counter() 

nn_model, optimizer, loss_function = nnAudio.build_model()

elapsed = time.perf_counter() - start 
print('>>> Elapsed %.3f seconds to build the model.' % elapsed)




start1 = time.perf_counter()

nnAudio.train_test(nn_model, optimizer, loss_function)


elapsed1 = time.perf_counter() - start1 
print('>>> Elapsed %.3f seconds to train and test.' % elapsed1)
print('>>>>>> Elapsed %.3f seconds in total.' % (elapsed+elapsed1))
