module     p0_ubaru_httbar_d43h9l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d43h9l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd43h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(32) :: acd43
      complex(ki) :: brack
      acd43(1)=dotproduct(k2,qshift)
      acd43(2)=dotproduct(qshift,spvak1k2)
      acd43(3)=abb43(12)
      acd43(4)=abb43(19)
      acd43(5)=dotproduct(qshift,qshift)
      acd43(6)=abb43(21)
      acd43(7)=dotproduct(qshift,spvak2l3)
      acd43(8)=abb43(18)
      acd43(9)=dotproduct(qshift,spval3k2)
      acd43(10)=abb43(15)
      acd43(11)=dotproduct(qshift,spval3l5)
      acd43(12)=abb43(16)
      acd43(13)=dotproduct(qshift,spval4l3)
      acd43(14)=abb43(14)
      acd43(15)=dotproduct(qshift,spval4l5)
      acd43(16)=abb43(13)
      acd43(17)=abb43(20)
      acd43(18)=abb43(28)
      acd43(19)=dotproduct(qshift,spvak1l3)
      acd43(20)=abb43(24)
      acd43(21)=dotproduct(qshift,spvak1l5)
      acd43(22)=abb43(17)
      acd43(23)=dotproduct(qshift,spval4k2)
      acd43(24)=abb43(10)
      acd43(25)=abb43(11)
      acd43(26)=acd43(3)*acd43(1)
      acd43(27)=acd43(8)*acd43(7)
      acd43(28)=acd43(10)*acd43(9)
      acd43(29)=acd43(12)*acd43(11)
      acd43(30)=acd43(14)*acd43(13)
      acd43(31)=acd43(16)*acd43(15)
      acd43(26)=-acd43(17)+acd43(31)+acd43(30)+acd43(29)+acd43(28)+acd43(27)+ac&
      &d43(26)
      acd43(26)=acd43(2)*acd43(26)
      acd43(27)=-acd43(4)*acd43(1)
      acd43(28)=acd43(6)*acd43(5)
      acd43(29)=-acd43(18)*acd43(9)
      acd43(30)=-acd43(20)*acd43(19)
      acd43(31)=-acd43(22)*acd43(21)
      acd43(32)=-acd43(24)*acd43(23)
      brack=acd43(25)+acd43(26)+acd43(27)+acd43(28)+acd43(29)+acd43(30)+acd43(3&
      &1)+acd43(32)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd43h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(39) :: acd43
      complex(ki) :: brack
      acd43(1)=k2(iv1)
      acd43(2)=dotproduct(qshift,spvak1k2)
      acd43(3)=abb43(12)
      acd43(4)=abb43(19)
      acd43(5)=qshift(iv1)
      acd43(6)=abb43(21)
      acd43(7)=spvak1k2(iv1)
      acd43(8)=dotproduct(k2,qshift)
      acd43(9)=dotproduct(qshift,spvak2l3)
      acd43(10)=abb43(18)
      acd43(11)=dotproduct(qshift,spval3k2)
      acd43(12)=abb43(15)
      acd43(13)=dotproduct(qshift,spval3l5)
      acd43(14)=abb43(16)
      acd43(15)=dotproduct(qshift,spval4l3)
      acd43(16)=abb43(14)
      acd43(17)=dotproduct(qshift,spval4l5)
      acd43(18)=abb43(13)
      acd43(19)=abb43(20)
      acd43(20)=spvak2l3(iv1)
      acd43(21)=spval3k2(iv1)
      acd43(22)=abb43(28)
      acd43(23)=spval3l5(iv1)
      acd43(24)=spval4l3(iv1)
      acd43(25)=spval4l5(iv1)
      acd43(26)=spvak1l3(iv1)
      acd43(27)=abb43(24)
      acd43(28)=spvak1l5(iv1)
      acd43(29)=abb43(17)
      acd43(30)=spval4k2(iv1)
      acd43(31)=abb43(10)
      acd43(32)=-acd43(3)*acd43(1)
      acd43(33)=-acd43(21)*acd43(12)
      acd43(34)=-acd43(20)*acd43(10)
      acd43(35)=-acd43(23)*acd43(14)
      acd43(36)=-acd43(24)*acd43(16)
      acd43(37)=-acd43(25)*acd43(18)
      acd43(32)=acd43(37)+acd43(36)+acd43(35)+acd43(34)+acd43(32)+acd43(33)
      acd43(32)=acd43(2)*acd43(32)
      acd43(33)=-acd43(8)*acd43(3)
      acd43(34)=-acd43(9)*acd43(10)
      acd43(35)=-acd43(11)*acd43(12)
      acd43(36)=-acd43(13)*acd43(14)
      acd43(37)=-acd43(15)*acd43(16)
      acd43(38)=-acd43(17)*acd43(18)
      acd43(33)=acd43(19)+acd43(38)+acd43(37)+acd43(36)+acd43(35)+acd43(34)+acd&
      &43(33)
      acd43(33)=acd43(7)*acd43(33)
      acd43(34)=acd43(4)*acd43(1)
      acd43(35)=acd43(6)*acd43(5)
      acd43(36)=acd43(22)*acd43(21)
      acd43(37)=acd43(27)*acd43(26)
      acd43(38)=acd43(29)*acd43(28)
      acd43(39)=acd43(31)*acd43(30)
      brack=acd43(32)+acd43(33)+acd43(34)-2.0_ki*acd43(35)+acd43(36)+acd43(37)+&
      &acd43(38)+acd43(39)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd43h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(29) :: acd43
      complex(ki) :: brack
      acd43(1)=d(iv1,iv2)
      acd43(2)=abb43(21)
      acd43(3)=k2(iv1)
      acd43(4)=spvak1k2(iv2)
      acd43(5)=abb43(12)
      acd43(6)=k2(iv2)
      acd43(7)=spvak1k2(iv1)
      acd43(8)=spvak2l3(iv2)
      acd43(9)=abb43(18)
      acd43(10)=spval3k2(iv2)
      acd43(11)=abb43(15)
      acd43(12)=spval3l5(iv2)
      acd43(13)=abb43(16)
      acd43(14)=spval4l3(iv2)
      acd43(15)=abb43(14)
      acd43(16)=spval4l5(iv2)
      acd43(17)=abb43(13)
      acd43(18)=spvak2l3(iv1)
      acd43(19)=spval3k2(iv1)
      acd43(20)=spval3l5(iv1)
      acd43(21)=spval4l3(iv1)
      acd43(22)=spval4l5(iv1)
      acd43(23)=acd43(3)*acd43(5)
      acd43(24)=acd43(18)*acd43(9)
      acd43(25)=acd43(19)*acd43(11)
      acd43(26)=acd43(20)*acd43(13)
      acd43(27)=acd43(21)*acd43(15)
      acd43(28)=acd43(22)*acd43(17)
      acd43(23)=acd43(28)+acd43(27)+acd43(26)+acd43(25)+acd43(24)+acd43(23)
      acd43(23)=acd43(4)*acd43(23)
      acd43(24)=acd43(6)*acd43(5)
      acd43(25)=acd43(8)*acd43(9)
      acd43(26)=acd43(10)*acd43(11)
      acd43(27)=acd43(12)*acd43(13)
      acd43(28)=acd43(14)*acd43(15)
      acd43(29)=acd43(16)*acd43(17)
      acd43(24)=acd43(29)+acd43(28)+acd43(27)+acd43(26)+acd43(25)+acd43(24)
      acd43(24)=acd43(7)*acd43(24)
      acd43(25)=acd43(2)*acd43(1)
      brack=acd43(23)+acd43(24)+2.0_ki*acd43(25)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd43h9_qp
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
      qshift = -k5
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
end module     p0_ubaru_httbar_d43h9l1d_qp
