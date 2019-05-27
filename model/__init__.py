def create_model(opt):
    if opt.model == 'MUNIT':
        from .MUNIT import MUNIT
        model = MUNIT()
        model.initialize(opt)
        model.setup()
    elif opt.model == 'MUNIT_semantic':
        from .MUNIT_semantic import MUNIT_semantic
        model = MUNIT_semantic()
        model.initialize(opt)
        model.setup()
    else:
        raise NotImplementedError('model [{}] is not found'.format(opt.model))

    print('model [{}] was created'.format(model.name()))
    return model
