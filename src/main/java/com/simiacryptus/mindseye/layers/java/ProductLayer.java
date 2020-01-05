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
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public @RefAware
class ProductLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);

  public ProductLayer() {
  }

  protected ProductLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static ProductLayer fromJson(@Nonnull final JsonObject json,
                                      Map<CharSequence, byte[]> rs) {
    return new ProductLayer(json);
  }

  public static @SuppressWarnings("unused")
  ProductLayer[] addRefs(ProductLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRef)
        .toArray((x) -> new ProductLayer[x]);
  }

  public static @SuppressWarnings("unused")
  ProductLayer[][] addRefs(ProductLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(ProductLayer::addRefs)
        .toArray((x) -> new ProductLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0];
    assert RefArrays.stream(inObj).mapToInt(x -> x.getData().length()).distinct()
        .count() == 1 : RefArrays.toString(
        RefArrays.stream(inObj).mapToInt(x -> x.getData().length()).toArray());
    @Nonnull final double[] sum_A = new double[in0.getData().length()];
    final Tensor[] outputA = RefIntStream.range(0, in0.getData().length())
        .mapToObj(dataIndex -> {
          double sum = 1;
          for (@Nonnull final Result input : inObj) {
            Tensor tensor = input.getData().get(dataIndex);
            @Nullable final double[] tensorData = tensor.getData();
            for (final double element2 : tensorData) {
              sum *= element2;
            }
          }
          sum_A[dataIndex] = sum;
          return new Tensor(new double[]{sum}, 1);
        }).toArray(i -> new Tensor[i]);
    return new Result(new TensorArray(outputA),
        new Result.Accumulator() {
          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList delta) {
            for (@Nonnull final Result input : inObj) {
              if (input.isAlive()) {
                TensorList data = input.getData();
                input.accumulate(buffer, new TensorArray(
                    RefIntStream.range(0, delta.length()).mapToObj(dataIndex -> {
                      Tensor dataTensor = delta.get(dataIndex);
                      Tensor lTensor = data.get(dataIndex);
                      @Nonnull final Tensor passback = new Tensor(lTensor.getDimensions());
                      for (int i = 0; i < lTensor.length(); i++) {
                        double d = lTensor.getData()[i];
                        double deltaV = dataTensor.get(0);
                        passback.set(i, d == 0 ? 0 : (deltaV * sum_A[dataIndex] / d));
                      }
                      return passback;
                    }).toArray(i -> new Tensor[i])));
              }
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
  public JsonObject getJson(Map<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
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
  ProductLayer addRef() {
    return (ProductLayer) super.addRef();
  }
}
