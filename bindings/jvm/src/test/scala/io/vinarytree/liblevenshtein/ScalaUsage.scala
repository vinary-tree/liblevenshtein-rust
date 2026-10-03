package io.vinarytree.liblevenshtein

import io.vinarytree.interop.DictionaryResource
import scala.jdk.CollectionConverters.*
import scala.util.Using

/** Compile-checked Scala 3 usage of the shipped Java API; no Scala facade is published. */
object ScalaUsage:
  def search(dictionary: DictionaryResource, input: String): Vector[Match] =
    Using.resource(new Transducer(dictionary, Algorithm.STANDARD)): transducer =>
      Using.resource(transducer.query(input, 2L, QueryOrder.DISTANCE_THEN_TERM)): cursor =>
        cursor.iterator().asScala.toVector
