module     p0_ubaru_httbar_d1h6l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d1h6l1d_qp.f90
   ! generator: buildfortran_d.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, private :: iv0
   integer, private :: iv1
   integer, private :: iv2
   real(ki), dimension(4), private :: qshift
   public :: derivative
contains
!---#[ function brack_1:
   pure function brack_1(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd1h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(23) :: acd1
      complex(ki) :: brack
      acd1(1)=dotproduct(k2,qshift)
      acd1(2)=dotproduct(qshift,spvak2k1)
      acd1(3)=abb1(10)
      acd1(4)=dotproduct(qshift,spval3k1)
      acd1(5)=abb1(30)
      acd1(6)=abb1(23)
      acd1(7)=abb1(9)
      acd1(8)=abb1(16)
      acd1(9)=dotproduct(qshift,spvak2l3)
      acd1(10)=dotproduct(qshift,spval5k1)
      acd1(11)=abb1(17)
      acd1(12)=abb1(31)
      acd1(13)=dotproduct(qshift,spvak2l4)
      acd1(14)=abb1(13)
      acd1(15)=abb1(22)
      acd1(16)=abb1(19)
      acd1(17)=abb1(20)
      acd1(18)=acd1(3)*acd1(2)
      acd1(19)=acd1(5)*acd1(4)
      acd1(18)=-acd1(6)+acd1(18)+acd1(19)
      acd1(18)=acd1(1)*acd1(18)
      acd1(19)=acd1(11)*acd1(9)
      acd1(20)=acd1(14)*acd1(13)
      acd1(19)=-acd1(15)+acd1(20)+acd1(19)
      acd1(19)=acd1(10)*acd1(19)
      acd1(20)=-acd1(7)*acd1(2)
      acd1(21)=-acd1(8)*acd1(4)
      acd1(22)=-acd1(12)*acd1(9)
      acd1(23)=-acd1(16)*acd1(13)
      brack=acd1(17)+acd1(18)+acd1(19)+acd1(20)+acd1(21)+acd1(22)+acd1(23)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd1h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(28) :: acd1
      complex(ki) :: brack
      acd1(1)=k2(iv1)
      acd1(2)=dotproduct(qshift,spvak2k1)
      acd1(3)=abb1(10)
      acd1(4)=dotproduct(qshift,spval3k1)
      acd1(5)=abb1(30)
      acd1(6)=abb1(23)
      acd1(7)=spvak2k1(iv1)
      acd1(8)=dotproduct(k2,qshift)
      acd1(9)=abb1(9)
      acd1(10)=spval3k1(iv1)
      acd1(11)=abb1(16)
      acd1(12)=spvak2l3(iv1)
      acd1(13)=dotproduct(qshift,spval5k1)
      acd1(14)=abb1(17)
      acd1(15)=abb1(31)
      acd1(16)=spval5k1(iv1)
      acd1(17)=dotproduct(qshift,spvak2l3)
      acd1(18)=dotproduct(qshift,spvak2l4)
      acd1(19)=abb1(13)
      acd1(20)=abb1(22)
      acd1(21)=spvak2l4(iv1)
      acd1(22)=abb1(19)
      acd1(23)=acd1(2)*acd1(3)
      acd1(24)=acd1(4)*acd1(5)
      acd1(23)=-acd1(6)+acd1(24)+acd1(23)
      acd1(23)=acd1(1)*acd1(23)
      acd1(24)=acd1(17)*acd1(14)
      acd1(25)=acd1(18)*acd1(19)
      acd1(24)=-acd1(20)+acd1(25)+acd1(24)
      acd1(24)=acd1(16)*acd1(24)
      acd1(25)=acd1(8)*acd1(3)
      acd1(25)=-acd1(9)+acd1(25)
      acd1(25)=acd1(7)*acd1(25)
      acd1(26)=acd1(8)*acd1(5)
      acd1(26)=-acd1(11)+acd1(26)
      acd1(26)=acd1(10)*acd1(26)
      acd1(27)=acd1(14)*acd1(13)
      acd1(27)=-acd1(15)+acd1(27)
      acd1(27)=acd1(12)*acd1(27)
      acd1(28)=acd1(19)*acd1(13)
      acd1(28)=-acd1(22)+acd1(28)
      acd1(28)=acd1(21)*acd1(28)
      brack=acd1(23)+acd1(24)+acd1(25)+acd1(26)+acd1(27)+acd1(28)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd1h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(21) :: acd1
      complex(ki) :: brack
      acd1(1)=k2(iv1)
      acd1(2)=spvak2k1(iv2)
      acd1(3)=abb1(10)
      acd1(4)=spval3k1(iv2)
      acd1(5)=abb1(30)
      acd1(6)=k2(iv2)
      acd1(7)=spvak2k1(iv1)
      acd1(8)=spval3k1(iv1)
      acd1(9)=spvak2l3(iv1)
      acd1(10)=spval5k1(iv2)
      acd1(11)=abb1(17)
      acd1(12)=spvak2l3(iv2)
      acd1(13)=spval5k1(iv1)
      acd1(14)=spvak2l4(iv2)
      acd1(15)=abb1(13)
      acd1(16)=spvak2l4(iv1)
      acd1(17)=acd1(2)*acd1(3)
      acd1(18)=acd1(4)*acd1(5)
      acd1(17)=acd1(17)+acd1(18)
      acd1(17)=acd1(1)*acd1(17)
      acd1(18)=acd1(7)*acd1(3)
      acd1(19)=acd1(8)*acd1(5)
      acd1(18)=acd1(19)+acd1(18)
      acd1(18)=acd1(6)*acd1(18)
      acd1(19)=acd1(9)*acd1(10)
      acd1(20)=acd1(12)*acd1(13)
      acd1(19)=acd1(20)+acd1(19)
      acd1(19)=acd1(11)*acd1(19)
      acd1(20)=acd1(14)*acd1(13)
      acd1(21)=acd1(16)*acd1(10)
      acd1(20)=acd1(21)+acd1(20)
      acd1(20)=acd1(15)*acd1(20)
      brack=acd1(17)+acd1(18)+acd1(19)+acd1(20)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd1h6_qp
      implicit none
      complex(ki), intent(in) :: mu2
      integer, intent(in), optional :: i1
      integer, intent(in), optional :: i2
      complex(ki) :: numerator
      complex(ki) :: loc
      integer :: t1
      integer :: deg
      complex(ki), dimension(4), parameter :: Q = (/ (0.0_ki,0.0_ki),(0.0_ki,0.&
      &0_ki),(0.0_ki,0.0_ki),(0.0_ki,0.0_ki)/)
      qshift = k5
      numerator = 0.0_ki
      deg = 0
      if(present(i1)) then
          iv1=i1
          deg=1
      else
          iv1=1
      end if
      if(present(i2)) then
          iv2=i2
          deg=2
      else
          iv2=1
      end if
      t1 = 0
      if(deg.eq.0) then
         numerator = cond(epspow.eq.t1,brack_1,Q,mu2)
         return
      end if
      if(deg.eq.1) then
         numerator = cond(epspow.eq.t1,brack_2,Q,mu2)
         return
      end if
      if(deg.eq.2) then
         numerator = cond(epspow.eq.t1,brack_3,Q,mu2)
         return
      end if
   end function derivative
!---#] function derivative:
end module     p0_ubaru_httbar_d1h6l1d_qp
