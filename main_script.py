import os




for i in range(100):
  print('\n\ntraining')
  exit = os.system('python mnist_ebgan_train.py')
  print('EXIT STATUS: {}'.format(exit))
  print('trained')
  print('generating')
  os.system('python mnist_ebgan_generate.py')
  print('generated')