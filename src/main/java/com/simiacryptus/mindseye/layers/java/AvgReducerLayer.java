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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.IntStream;

@SuppressWarnings("serial")
public class AvgReducerLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);

  public AvgReducerLayer() {
  }

  protected AvgReducerLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  @SuppressWarnings("unused")
  public static AvgReducerLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AvgReducerLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    return new Result(
        new TensorArray(IntStream.range(0, inObj[0].getData().length()).parallel().mapToDouble(dataIndex -> {
          double sum = 0;
          for (@Nonnull final Result element : inObj) {
            Tensor tensor = element.getData().get(dataIndex);
            @Nullable final double[] input = tensor.getData();
            for (final double element2 : input) {
              sum += element2 / input.length;
            }
          }
          return sum;
        }).mapToObj(x -> new Tensor(new double[]{x}, new int[]{1})).toArray(i -> new Tensor[i])),
        (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList delta) -> {
          for (@Nonnull final Result in_l : inObj) {
            if (in_l.isAlive()) {
              TensorList inData = in_l.getData();
              @Nonnull final TensorList tensorList = new TensorArray(IntStream.range(0, inData.length()).parallel().mapToObj(dataIndex -> {
                Tensor deltaTensor = delta.get(dataIndex);
                final double deltaV = deltaTensor.get(0);
                @Nonnull final Tensor passback = new Tensor(inData.getDimensions());
                final int dim = passback.length();
                for (int i = 0; i < dim; i++) {
                  passback.set(i, deltaV / dim);
                }
                return passback;
              }).toArray(i -> new Tensor[i]));
              in_l.accumulate(buffer, tensorList);
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

      @Override
      protected void _free() {
      }

    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
