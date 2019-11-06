import time
from os import environ
from functools import wraps
from datetime import timedelta, datetime
import tensorflow as tf
from tsaplay.utils.io import cprnt


def tf_class_distribution(labels):
    with tf.name_scope("debug_distribution_monitor"):
        ones = tf.ones_like(labels)
        zeros = tf.zeros_like(labels)

        negative = tf.reduce_sum(tf.where(tf.equal(labels, 0), ones, zeros))
        neutral = tf.reduce_sum(tf.where(tf.equal(labels, 1), ones, zeros))
        positive = tf.reduce_sum(tf.where(tf.equal(labels, 2), ones, zeros))

        distribution = tf.stack([negative, neutral, positive])
        totals = tf.reduce_sum(ones) * tf.ones_like(distribution)

        percentages = tf.divide(distribution, totals) * 100

    return tf.cast(percentages, tf.int32)


def timeit(pre="", post=""):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            if environ.get("TIMEIT", "ON").lower() == "off":
                return func(*args, **kw)
            name = func.__qualname__ + "():"
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cprnt(c=time_stamp, r=name, g=pre)
            start_time = time.time()
            result = func(*args, **kw)
            end_time = time.time()
            time_taken = timedelta(seconds=(end_time - start_time))
            time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if post:
                cprnt(
                    c=time_stamp, r=name, g=post + " in", row=str(time_taken)
                )
            return result

        return wrapper

    return inner_decorator
