import LeafNN as leaf
import time



start = time.perf_counter() 

leaf_model = leaf.build_model()

elapsed = time.perf_counter() - start 
print('>>> Elapsed %.3f seconds to build the model.' % elapsed)



start1 = time.perf_counter()

test_ds = leaf.train_model(leaf_model)

elapsed1 = time.perf_counter() - start1 
print('>>> Elapsed %.3f seconds to train.' % elapsed1)




start2 = time.perf_counter()

leaf.test_model(test_ds, leaf_model)

elapsed2 = time.perf_counter() - start2
print('>>> Elapsed %.3f seconds to test.' % elapsed2)
print('>>>>>> Elapsed %.3f seconds in total.' % (elapsed+elapsed1+elapsed2))

