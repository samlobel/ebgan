import os



print('generating before we train')
os.system('python mnist_ebgan_generate.py')


for i in range(100):
  print('\n\ntraining')
  exit_status = os.system('python mnist_ebgan_train.py')
  print('EXIT STATUS: {}'.format(exit_status))
  print('trained')
  print('generating')
  os.system('python mnist_ebgan_generate.py')
  print('generated')