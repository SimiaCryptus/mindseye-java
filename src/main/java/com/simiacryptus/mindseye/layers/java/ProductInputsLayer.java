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

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class ProductInputsLayer extends LayerBase {

  public ProductInputsLayer() {
  }

  protected ProductInputsLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static ProductInputsLayer fromJson(@Nonnull final JsonObject json,
                                            com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new ProductInputsLayer(json);
  }

  public static @SuppressWarnings("unused")
  ProductInputsLayer[] addRefs(ProductInputsLayer[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayer::addRef)
        .toArray((x) -> new ProductInputsLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ProductInputsLayer[][] addRefs(ProductInputsLayer[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(ProductInputsLayer::addRefs)
        .toArray((x) -> new ProductInputsLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert inObj.length > 1;
    for (int i = 1; i < inObj.length; i++) {
      final int dim0 = Tensor.length(inObj[0].getData().getDimensions());
      final int dimI = Tensor.length(inObj[i].getData().getDimensions());
      if (dim0 != 1 && dimI != 1 && dim0 != dimI) {
        throw new IllegalArgumentException(
            com.simiacryptus.ref.wrappers.RefArrays.toString(inObj[0].getData().getDimensions()) + " != "
                + com.simiacryptus.ref.wrappers.RefArrays.toString(inObj[i].getData().getDimensions()));
      }
    }
    return new Result(
        com.simiacryptus.ref.wrappers.RefArrays.stream(inObj).parallel().map(x -> x.getData()).reduce((l, r) -> {
          return new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream.range(0, Math.max(l.length(), r.length()))
              .parallel().mapToObj(i1 -> {
                @Nullable final Tensor left = l.get(1 == l.length() ? 0 : i1);
                @Nullable final Tensor right = r.get(1 == r.length() ? 0 : i1);
                return Tensor.product(left, right);
              }).toArray(i -> new Tensor[i]));
        }).get(), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
      for (@Nonnull final Result input : inObj) {
        if (input.isAlive()) {
          @Nonnull
          TensorList passback = com.simiacryptus.ref.wrappers.RefArrays.stream(inObj).parallel().map(x -> {
            return x == input ? delta : x.getData();
          }).reduce((l, r) -> {
            return new TensorArray(com.simiacryptus.ref.wrappers.RefIntStream
                .range(0, Math.max(l.length(), r.length())).parallel().mapToObj(j -> {
                  @Nullable final Tensor left = l.get(1 == l.length() ? 0 : j);
                  @Nullable final Tensor right = r.get(1 == r.length() ? 0 : j);
                  return Tensor.product(left, right);
                }).toArray(j -> new Tensor[j]));
          }).get();
          final TensorList inputData = input.getData();
          if (1 == inputData.length() && 1 < passback.length()) {
            passback = new TensorArray(passback.stream().reduce((a, b) -> {
              return a.addAndFree(b);
            }).get());
          }
          if (1 == Tensor.length(inputData.getDimensions()) && 1 < Tensor.length(passback.getDimensions())) {
            passback = new TensorArray(passback.stream().map((a) -> {
              return new Tensor(a.sum());
            }).toArray(i -> new Tensor[i]));
          }
          input.accumulate(buffer, passback);
        }
      }
    }) {

      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }

      public void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public com.simiacryptus.ref.wrappers.RefList<double[]> state() {
    return com.simiacryptus.ref.wrappers.RefArrays.asList();
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
