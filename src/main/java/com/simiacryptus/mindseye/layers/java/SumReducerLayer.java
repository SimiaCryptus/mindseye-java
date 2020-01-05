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
class SumReducerLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);

  public SumReducerLayer() {
  }

  protected SumReducerLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static SumReducerLayer fromJson(@Nonnull final JsonObject json,
                                         Map<CharSequence, byte[]> rs) {
    return new SumReducerLayer(json);
  }

  public static @SuppressWarnings("unused")
  SumReducerLayer[] addRefs(SumReducerLayer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayer::addRef)
        .toArray((x) -> new SumReducerLayer[x]);
  }

  public static @SuppressWarnings("unused")
  SumReducerLayer[][] addRefs(SumReducerLayer[][] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SumReducerLayer::addRefs)
        .toArray((x) -> new SumReducerLayer[x][]);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    return new Result(new TensorArray(RefIntStream.range(0, inObj[0].getData().length())
        .parallel().mapToDouble(dataIndex -> {
          double sum = 0;
          for (@Nonnull final Result element : inObj) {
            @Nullable
            Tensor tensor = element.getData().get(dataIndex);
            @Nullable final double[] input = tensor.getData();
            for (final double element2 : input) {
              sum += element2;
            }
          }
          return sum;
        }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])),
        new Result.Accumulator() {
          @Override
          public void accept(DeltaSet<UUID> buffer, TensorList data) {
            for (@Nonnull final Result in_l : inObj) {
              if (in_l.isAlive()) {
                @Nonnull
                TensorArray tensorArray = new TensorArray(RefIntStream
                    .range(0, in_l.getData().length()).parallel().mapToObj(dataIndex -> {
                      Tensor tensor = data.get(dataIndex);
                      assert 1 == tensor.length() : RefArrays
                          .toString(tensor.getDimensions());
                      @Nonnull final Tensor passback = new Tensor(in_l.getData().getDimensions());
                      for (int i = 0; i < Tensor.length(in_l.getData().getDimensions()); i++) {
                        passback.set(i, tensor.get(0));
                      }
                      return passback;
                    }).toArray(i -> new Tensor[i]));
                in_l.accumulate(buffer, tensorArray);
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
  SumReducerLayer addRef() {
    return (SumReducerLayer) super.addRef();
  }
}
