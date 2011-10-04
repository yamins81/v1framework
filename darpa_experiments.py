from starflow.protocols import Apply, Applies, protocolize

import deploy
import pythor_protocols as protocols

@Applies(deploy.images,args=('../config/darpa_heli_200_2500.py',False,('../../darpa/helidata/',)))
def generate_darpa_images():
    Apply()


@protocolize()
def make_darpa_test_models(depends_on='../config/darpa_test_models.py'):
    """
    """
    protocols.model_protocol(depends_on,parallel=False,write=True)
            

@protocolize()
def extract_darpa_test_models(depends_on=('../config/darpa_test_extraction.py',
                                          '../config/darpa_test_models.py',
                                          '../config/darpa_heli_200_2500.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,parallel=True,save_to_db=True,batch_size=100)



@Applies(deploy.images,args=('../config/darpa_heli_200_2500_enriched.py',False,('../../darpa/helidata/',)))
def generate_darpa_enriched_images():
    Apply()


@protocolize()
def make_darpa_models(depends_on='../config/darpa_models.py'):
    protocols.model_protocol(depends_on,parallel=False,write=True)

@protocolize()
def extract_darpa_models(depends_on=('../config/darpa_extraction.py',
                                     '../config/darpa_models.py',
                                     '../config/darpa_heli_200_2500_enriched.py')):
    a,b,c = depends_on
    protocols.extraction_protocol(a,b,c,convolve_func_name='numpy', write=True,
                                  parallel=True, save_to_db=True, batch_size=100)
