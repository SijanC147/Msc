import time
import pprint
from os import environ
from functools import wraps
from datetime import timedelta, datetime
from termcolor import colored
import tensorflow as tf


def color(key):
    return {
        "r": "red",
        "g": "green",
        "y": "yellow",
        "b": "blue",
        "m": "magenta",
        "c": "cyan",
        "w": "white",
    }.get(key, "grey")


def cprnt(*args, **kwargs):
    output = ""
    for arg in args:
        kwargs.update({"w": arg})
    for (color_key, string) in kwargs.items():
        if color_key == "end":
            continue
        if not isinstance(string, str):
            string = pprint.pformat(string)
        col = "".join(filter(str.isalpha, color_key))
        index = col.find("o")
        if index != -1:
            txt, _, frgnd = col.partition("o")
            output += colored(string, color(txt), "on_" + color(frgnd)) + " "
        else:
            output += colored(string, color(col)) + " "
    print(output, end=kwargs.get("end", "\n"))


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
            cprnt(c=time_stamp, r=name, g=post + " in", row=str(time_taken))
            return result

        return wrapper

    return inner_decorator
