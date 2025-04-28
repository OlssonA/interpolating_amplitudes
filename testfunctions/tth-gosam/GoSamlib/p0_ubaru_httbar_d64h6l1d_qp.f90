module     p0_ubaru_httbar_d64h6l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity6d64h6l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd64h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(35) :: acd64
      complex(ki) :: brack
      acd64(1)=dotproduct(k2,qshift)
      acd64(2)=abb64(13)
      acd64(3)=dotproduct(qshift,qshift)
      acd64(4)=abb64(12)
      acd64(5)=dotproduct(qshift,spvak2k1)
      acd64(6)=dotproduct(qshift,spval5l3)
      acd64(7)=abb64(8)
      acd64(8)=dotproduct(qshift,spval5l4)
      acd64(9)=abb64(11)
      acd64(10)=abb64(15)
      acd64(11)=abb64(19)
      acd64(12)=abb64(20)
      acd64(13)=dotproduct(qshift,spvak2l3)
      acd64(14)=abb64(14)
      acd64(15)=dotproduct(qshift,spvak2l4)
      acd64(16)=abb64(7)
      acd64(17)=dotproduct(qshift,spvak2l5)
      acd64(18)=abb64(23)
      acd64(19)=dotproduct(qshift,spval3k1)
      acd64(20)=abb64(9)
      acd64(21)=dotproduct(qshift,spval3l5)
      acd64(22)=abb64(16)
      acd64(23)=dotproduct(qshift,spval5k1)
      acd64(24)=abb64(10)
      acd64(25)=acd64(7)*acd64(6)
      acd64(26)=acd64(9)*acd64(8)
      acd64(25)=-acd64(10)+acd64(26)+acd64(25)
      acd64(25)=acd64(5)*acd64(25)
      acd64(26)=-acd64(2)*acd64(1)
      acd64(27)=acd64(4)*acd64(3)
      acd64(28)=-acd64(11)*acd64(6)
      acd64(29)=-acd64(12)*acd64(8)
      acd64(30)=-acd64(14)*acd64(13)
      acd64(31)=-acd64(16)*acd64(15)
      acd64(32)=-acd64(18)*acd64(17)
      acd64(33)=-acd64(20)*acd64(19)
      acd64(34)=-acd64(22)*acd64(21)
      acd64(35)=-acd64(24)*acd64(23)
      brack=acd64(25)+acd64(26)+acd64(27)+acd64(28)+acd64(29)+acd64(30)+acd64(3&
      &1)+acd64(32)+acd64(33)+acd64(34)+acd64(35)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd64h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(38) :: acd64
      complex(ki) :: brack
      acd64(1)=k2(iv1)
      acd64(2)=abb64(13)
      acd64(3)=qshift(iv1)
      acd64(4)=abb64(12)
      acd64(5)=spvak2k1(iv1)
      acd64(6)=dotproduct(qshift,spval5l3)
      acd64(7)=abb64(8)
      acd64(8)=dotproduct(qshift,spval5l4)
      acd64(9)=abb64(11)
      acd64(10)=abb64(15)
      acd64(11)=spval5l3(iv1)
      acd64(12)=dotproduct(qshift,spvak2k1)
      acd64(13)=abb64(19)
      acd64(14)=spval5l4(iv1)
      acd64(15)=abb64(20)
      acd64(16)=spvak2l3(iv1)
      acd64(17)=abb64(14)
      acd64(18)=spvak2l4(iv1)
      acd64(19)=abb64(7)
      acd64(20)=spvak2l5(iv1)
      acd64(21)=abb64(23)
      acd64(22)=spval3k1(iv1)
      acd64(23)=abb64(9)
      acd64(24)=spval3l5(iv1)
      acd64(25)=abb64(16)
      acd64(26)=spval5k1(iv1)
      acd64(27)=abb64(10)
      acd64(28)=-acd64(6)*acd64(7)
      acd64(29)=-acd64(8)*acd64(9)
      acd64(28)=acd64(10)+acd64(29)+acd64(28)
      acd64(28)=acd64(5)*acd64(28)
      acd64(29)=-acd64(12)*acd64(7)
      acd64(29)=acd64(13)+acd64(29)
      acd64(29)=acd64(11)*acd64(29)
      acd64(30)=-acd64(12)*acd64(9)
      acd64(30)=acd64(15)+acd64(30)
      acd64(30)=acd64(14)*acd64(30)
      acd64(31)=acd64(2)*acd64(1)
      acd64(32)=acd64(4)*acd64(3)
      acd64(33)=acd64(17)*acd64(16)
      acd64(34)=acd64(19)*acd64(18)
      acd64(35)=acd64(21)*acd64(20)
      acd64(36)=acd64(23)*acd64(22)
      acd64(37)=acd64(25)*acd64(24)
      acd64(38)=acd64(27)*acd64(26)
      brack=acd64(28)+acd64(29)+acd64(30)+acd64(31)-2.0_ki*acd64(32)+acd64(33)+&
      &acd64(34)+acd64(35)+acd64(36)+acd64(37)+acd64(38)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd64h6_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(13) :: acd64
      complex(ki) :: brack
      acd64(1)=d(iv1,iv2)
      acd64(2)=abb64(12)
      acd64(3)=spvak2k1(iv1)
      acd64(4)=spval5l3(iv2)
      acd64(5)=abb64(8)
      acd64(6)=spval5l4(iv2)
      acd64(7)=abb64(11)
      acd64(8)=spvak2k1(iv2)
      acd64(9)=spval5l3(iv1)
      acd64(10)=spval5l4(iv1)
      acd64(11)=acd64(4)*acd64(5)
      acd64(12)=acd64(6)*acd64(7)
      acd64(11)=acd64(12)+acd64(11)
      acd64(11)=acd64(3)*acd64(11)
      acd64(12)=acd64(9)*acd64(5)
      acd64(13)=acd64(10)*acd64(7)
      acd64(12)=acd64(13)+acd64(12)
      acd64(12)=acd64(8)*acd64(12)
      acd64(13)=acd64(2)*acd64(1)
      brack=acd64(11)+acd64(12)+2.0_ki*acd64(13)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd64h6_qp
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
      qshift = k2
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
end module     p0_ubaru_httbar_d64h6l1d_qp
