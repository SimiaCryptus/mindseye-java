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
import com.simiacryptus.lang.ref.ReferenceCountingBase;
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
public class BiasMetaLayer extends LayerBase {


  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasMetaLayer.class);

  public BiasMetaLayer() {
  }

  protected BiasMetaLayer(@Nonnull final JsonObject id) {
    super(id);
  }

  public static BiasMetaLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new BiasMetaLayer(json);
  }

  @Nullable
  @Override
  public Result evalAndFree(@Nonnull final Result... inObj) {
    final Result in0 = inObj[0];
    final Result in1 = inObj[1];
    TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    final TensorList data1 = in1.getData();
    Tensor tensor1 = data1.get(0);
    final Tensor[] tensors = IntStream.range(0, itemCnt)
        .parallel()
        .mapToObj(dataIndex -> {
          Tensor tensor = data0.get(dataIndex);
          Tensor mapIndex = tensor.mapIndex((v, c) -> {
            return v + tensor1.get(c);
          });
          tensor.freeRef();
          return mapIndex;
        })
        .toArray(i -> new Tensor[i]);
    tensor1.freeRef();
    data1.freeRef();
    data0.freeRef();
    Tensor tensor0 = tensors[0].addRef();
    return new Result(TensorArray.wrap(tensors), (@Nonnull final DeltaSet<UUID> buffer, @Nonnull final TensorList data) -> {
      if (in1.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, data.length())
            .mapToObj(i -> {
              if (i == 0) return tensor0.mapCoords((c) -> {
                return IntStream.range(0, itemCnt).mapToDouble(j -> {
                  Tensor tensor = data.get(j);
                  double v1 = tensor.get(c);
                  tensor.freeRef();
                  return v1;
                }).sum();
              });
              else {
                return tensor0.mapCoords(v -> 0);
              }
            }).toArray(i -> new Tensor[i]));
        in1.accumulate(buffer, tensorArray);
      }
      if (in0.isAlive()) {
        in0.accumulate(buffer, data);
      } else {
        data.freeRef();
      }
    }) {

      @Override
      protected void _free() {
        tensor0.freeRef();
        Arrays.stream(inObj).forEach(ReferenceCountingBase::freeRef);
      }

      @Override
      public boolean isAlive() {
        return in0.isAlive() || in1.isAlive();
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
