# Optimizer Hook Configuration
optimizer_config = dict(
    grad_clip=dict(
        max_norm=50, 
        norm_type=2,
        ),
    bmuf_config=dict(
        warmup_steps=8600,
        use_nesterov=True,
        block=50,
        block_lr=1.0,
        block_momentum = 0.85, # 1.0-1.0/num_workers
    )
 )
log_level = 'INFO'
log_config = dict(
    interval=50,  # log at every 50 iterations
    reset_flag=True,
    hooks=[
        #dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook'),
    ])

load_from = None
resume_from = None