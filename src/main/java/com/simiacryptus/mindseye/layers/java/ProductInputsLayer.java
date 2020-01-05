/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefAware;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.IntFunction;

@SuppressWarnings("serial")
public @RefAware
class ProductInputsLayer extends LayerBase {

  public ProductInputsLayer() {
  }

  protected ProductInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static ProductInputsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ProductInputsLayer(json);
  }

  public static @SuppressWarnings("unused")
  ProductInputsLayer[] addRefs(ProductInputsLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayer::addRef)
        .toArray((x) -> new ProductInputsLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ProductInputsLayer[][] addRefs(ProductInputsLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayer::addRefs)
        .toArray((x) -> new ProductInputsLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length > 1;
    for (int i = 1; i < inObj.length; i++) {
      TensorList temp_57_0011 = inObj[0].getData();
      final int dim0 = Tensor.length(temp_57_0011.getDimensions());
      if (null != temp_57_0011)
        temp_57_0011.freeRef();
      TensorList temp_57_0012 = inObj[i].getData();
      final int dimI = Tensor.length(temp_57_0012.getDimensions());
      if (null != temp_57_0012)
        temp_57_0012.freeRef();
      if (dim0 != 1 && dimI != 1 && dim0 != dimI) {
        TensorList temp_57_0013 = inObj[0].getData();
        TensorList temp_57_0014 = inObj[i].getData();
        IllegalArgumentException temp_57_0010 = new IllegalArgumentException(
            RefArrays.toString(temp_57_0013.getDimensions()) + " != "
                + RefArrays.toString(temp_57_0014.getDimensions()));
        if (null != temp_57_0014)
          temp_57_0014.freeRef();
        if (null != temp_57_0013)
          temp_57_0013.freeRef();
        if (null != inObj)
          ReferenceCounting.freeRefs(inObj);
        throw temp_57_0010;
      }
    }
    try {
      return new Result(RefArrays.stream(Result.addRefs(inObj)).parallel().map(x -> {
        TensorList temp_57_0001 = x.getData();
        if (null != x)
          x.freeRef();
        return temp_57_0001;
      }).reduce((l, r) -> {
        TensorArray temp_57_0002 = new TensorArray(RefIntStream
            .range(0, Math.max(l.length(), r.length())).parallel().mapToObj(RefUtil
                .wrapInterface((IntFunction<? extends Tensor>) i1 -> {
                  @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i1);
                  @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i1);
                  Tensor temp_57_0003 = Tensor
                      .product(left == null ? null : left.addRef(), right == null ? null : right.addRef());
                  if (null != right)
                    right.freeRef();
                  if (null != left)
                    left.freeRef();
                  return temp_57_0003;
                }, l == null ? null : l.addRef(), r == null ? null : r.addRef()))
            .toArray(i -> new Tensor[i]));
        if (null != r)
          r.freeRef();
        if (null != l)
          l.freeRef();
        return temp_57_0002;
      }).get(), new Result.Accumulator() {
        {
          Result.addRefs(inObj);
        }

        @Override
        public void accept(DeltaSet<UUID> buffer, TensorList delta) {
          for (@Nonnull final Result input : inObj) {
            if (input.isAlive()) {
              @Nonnull
              TensorList passback = RefArrays.stream(Result.addRefs(inObj)).parallel()
                  .map(RefUtil.wrapInterface(
                      (Function<? super Result, ? extends TensorList>) x -> {
                        TensorList temp_57_0004 = x == input ? delta : x.getData();
                        if (null != x)
                          x.freeRef();
                        return temp_57_0004;
                      }, delta == null ? null : delta.addRef(), input == null ? null : input.addRef()))
                  .reduce((l, r) -> {
                    TensorArray temp_57_0005 = new TensorArray(
                        RefIntStream.range(0, Math.max(l.length(), r.length())).parallel()
                            .mapToObj(RefUtil.wrapInterface(
                                (IntFunction<? extends Tensor>) j -> {
                                  @Nullable final Tensor left = l.get(1 == l.length() ? 0 : j);
                                  @Nullable final Tensor right = r.get(1 == r.length() ? 0 : j);
                                  Tensor temp_57_0006 = Tensor.product(
                                      left == null ? null : left.addRef(), right == null ? null : right.addRef());
                                  if (null != right)
                                    right.freeRef();
                                  if (null != left)
                                    left.freeRef();
                                  return temp_57_0006;
                                }, l == null ? null : l.addRef(), r == null ? null : r.addRef()))
                            .toArray(j -> new Tensor[j]));
                    if (null != r)
                      r.freeRef();
                    if (null != l)
                      l.freeRef();
                    return temp_57_0005;
                  }).get();
              final TensorList inputData = input.getData();
              if (1 == inputData.length() && 1 < passback.length()) {
                passback = new TensorArray(passback.stream().reduce((a, b) -> {
                  Tensor temp_57_0007 = a.addAndFree(b == null ? null : b.addRef());
                  if (null != b)
                    b.freeRef();
                  if (null != a)
                    a.freeRef();
                  return temp_57_0007;
                }).get());
              }
              if (1 == Tensor.length(inputData.getDimensions()) && 1 < Tensor.length(passback.getDimensions())) {
                passback = new TensorArray(passback.stream().map((a) -> {
                  Tensor temp_57_0008 = new Tensor(a.sum());
                  if (null != a)
                    a.freeRef();
                  return temp_57_0008;
                }).toArray(i -> new Tensor[i]));
              }
              if (null != inputData)
                inputData.freeRef();
              input.accumulate(buffer == null ? null : buffer.addRef(), passback == null ? null : passback);
            }
          }
          if (null != delta)
            delta.freeRef();
          if (null != buffer)
            buffer.freeRef();
        }

        public @SuppressWarnings("unused")
        void _free() {
          ReferenceCounting.freeRefs(inObj);
        }
      }) {

        {
          Result.addRefs(inObj);
        }

        @Override
        public boolean isAlive() {
          for (@Nonnull final Result element : inObj)
            if (element.isAlive()) {
              return true;
            }
          return false;
        }

        public void _free() {
          ReferenceCounting.freeRefs(inObj);
        }

      };
    } finally {
      ReferenceCounting.freeRefs(inObj);
    }
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  ProductInputsLayer addRef() {
    return (ProductInputsLayer) super.addRef();
  }
}
