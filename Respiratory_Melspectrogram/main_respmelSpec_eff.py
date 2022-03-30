import eff_respmelSpec as melSpec
import time


############ BUILD MODEL #################
start = time.perf_counter() 

melSpec_model = melSpec.build_model()

elapsed = time.perf_counter() - start 
print('>>> Elapsed %.3f seconds to build the model.' % elapsed)




############ TRAIN MODEL #################
start1 = time.perf_counter()

test_ds = melSpec.train_model(melSpec_model)

elapsed1 = time.perf_counter() - start1 
print('>>> Elapsed %.3f seconds to train.' % elapsed1)





############ TEST MODEL #################
start2 = time.perf_counter()

melSpec.test_model(test_ds, melSpec_model)

elapsed2 = time.perf_counter() - start2
print('>>> Elapsed %.3f seconds to test.' % elapsed2)
print('>>>>>> Elapsed %.3f seconds in total.' % (elapsed+elapsed1+elapsed2))
