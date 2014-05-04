import os
import sys

from setuptools import setup, find_packages, Command

import sklearn_extra

def get_reqs(testing=False):
    return [
        'numpy',
        'scipy',
        'scikit-learn>=0.14.1',
        'six',
    ]

class TestCommand(Command):
    description = "Runs unittests."
    
    user_options = [
        ('name=', None,
         'Name of the specific test to run.'),
        ('virtual-env-dir=', None,
         'The location of the virtual environment to use.'),
        ('pv=', None,
         'The version of Python to use. e.g. 2.7 or 3'),
    ]
    
    def initialize_options(self):
        self.name = None
        self.virtual_env_dir = './.env%s'
        self.pv = 2.7
        
    def finalize_options(self):
        pass
    
    def build_virtualenv(self, pv):
        #print('pv=',self.pv)
        virtual_env_dir = self.virtual_env_dir % self.pv
        kwargs = dict(virtual_env_dir=virtual_env_dir, pv=self.pv)
        if not os.path.isdir(virtual_env_dir):
            cmd = 'virtualenv -p /usr/bin/python{pv} {virtual_env_dir}'.format(**kwargs)
            #print(cmd)
            os.system(cmd)
            
            cmd = '. {virtual_env_dir}/bin/activate; easy_install -U distribute; deactivate'.format(**kwargs)
            os.system(cmd)
            
            for package in get_reqs(testing=True):
                kwargs['package'] = package
                cmd = '. {virtual_env_dir}/bin/activate; pip install -U {package}; deactivate'.format(**kwargs)
                #print(cmd)
                os.system(cmd)
    
    def run(self):
        self.build_virtualenv(self.pv)
        kwargs = dict(pv=self.pv, name=self.name)
        if self.name:
            cmd = '. ./.env{pv}/bin/activate; python sklearn_extra/tests.py; deactivate'.format(**kwargs)
        else:
            cmd = '. ./.env{pv}/bin/activate; python sklearn_extra/tests.py; deactivate'.format(**kwargs)
        #print(cmd)
        os.system(cmd)

setup(name='sklearn_extra',
    version=sklearn_extra.__version__,
    description='Additional classification and regression algorithms based on the scikit-learn library.',
    author='Chris Spencer',
    author_email='chrisspen@gmail.com',
    url='https://github.com/chrisspen/sklearn_extra',
    license='Apache License',
    packages = find_packages(),
    install_requires=get_reqs(),
    zip_safe=True,
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    platforms=['OS Independent'],
    cmdclass={
        'test': TestCommand,
    },
)
