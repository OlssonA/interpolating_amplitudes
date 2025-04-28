module     p0_ubaru_httbar_d77h1l1d_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d77h1l1d_qp.f90
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
      use p0_ubaru_httbar_abbrevd77h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(26) :: acd77
      complex(ki) :: brack
      acd77(1)=dotproduct(qshift,qshift)
      acd77(2)=abb77(13)
      acd77(3)=dotproduct(qshift,spvak1k2)
      acd77(4)=dotproduct(qshift,spval3k2)
      acd77(5)=abb77(10)
      acd77(6)=dotproduct(qshift,spval4k2)
      acd77(7)=abb77(21)
      acd77(8)=dotproduct(qshift,spval4l3)
      acd77(9)=abb77(20)
      acd77(10)=dotproduct(qshift,spval5k2)
      acd77(11)=abb77(18)
      acd77(12)=dotproduct(qshift,spval5l3)
      acd77(13)=abb77(15)
      acd77(14)=abb77(12)
      acd77(15)=dotproduct(qshift,spvak1l3)
      acd77(16)=abb77(16)
      acd77(17)=abb77(24)
      acd77(18)=abb77(26)
      acd77(19)=abb77(22)
      acd77(20)=abb77(11)
      acd77(21)=abb77(19)
      acd77(22)=acd77(5)*acd77(4)
      acd77(23)=acd77(7)*acd77(6)
      acd77(24)=acd77(9)*acd77(8)
      acd77(25)=acd77(11)*acd77(10)
      acd77(26)=acd77(13)*acd77(12)
      acd77(22)=-acd77(14)+acd77(26)+acd77(25)+acd77(24)+acd77(23)+acd77(22)
      acd77(22)=acd77(3)*acd77(22)
      acd77(23)=acd77(16)*acd77(6)
      acd77(24)=acd77(18)*acd77(10)
      acd77(23)=-acd77(20)+acd77(24)+acd77(23)
      acd77(23)=acd77(15)*acd77(23)
      acd77(24)=acd77(2)*acd77(1)
      acd77(25)=-acd77(17)*acd77(6)
      acd77(26)=-acd77(19)*acd77(10)
      brack=acd77(21)+acd77(22)+acd77(23)+acd77(24)+acd77(25)+acd77(26)
   end function brack_1
!---#] function brack_1:
!---#[ function brack_2:
   pure function brack_2(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(33) :: acd77
      complex(ki) :: brack
      acd77(1)=qshift(iv1)
      acd77(2)=abb77(13)
      acd77(3)=spvak1k2(iv1)
      acd77(4)=dotproduct(qshift,spval3k2)
      acd77(5)=abb77(10)
      acd77(6)=dotproduct(qshift,spval4k2)
      acd77(7)=abb77(21)
      acd77(8)=dotproduct(qshift,spval4l3)
      acd77(9)=abb77(20)
      acd77(10)=dotproduct(qshift,spval5k2)
      acd77(11)=abb77(18)
      acd77(12)=dotproduct(qshift,spval5l3)
      acd77(13)=abb77(15)
      acd77(14)=abb77(12)
      acd77(15)=spval3k2(iv1)
      acd77(16)=dotproduct(qshift,spvak1k2)
      acd77(17)=spval4k2(iv1)
      acd77(18)=dotproduct(qshift,spvak1l3)
      acd77(19)=abb77(16)
      acd77(20)=abb77(24)
      acd77(21)=spval4l3(iv1)
      acd77(22)=spval5k2(iv1)
      acd77(23)=abb77(26)
      acd77(24)=abb77(22)
      acd77(25)=spval5l3(iv1)
      acd77(26)=spvak1l3(iv1)
      acd77(27)=abb77(11)
      acd77(28)=-acd77(13)*acd77(25)
      acd77(29)=-acd77(9)*acd77(21)
      acd77(30)=-acd77(5)*acd77(15)
      acd77(31)=-acd77(22)*acd77(11)
      acd77(32)=-acd77(17)*acd77(7)
      acd77(28)=acd77(32)+acd77(31)+acd77(30)+acd77(28)+acd77(29)
      acd77(28)=acd77(16)*acd77(28)
      acd77(29)=-acd77(13)*acd77(12)
      acd77(30)=-acd77(10)*acd77(11)
      acd77(31)=-acd77(9)*acd77(8)
      acd77(32)=-acd77(6)*acd77(7)
      acd77(33)=-acd77(5)*acd77(4)
      acd77(29)=acd77(33)+acd77(32)+acd77(31)+acd77(30)+acd77(14)+acd77(29)
      acd77(29)=acd77(3)*acd77(29)
      acd77(30)=-acd77(10)*acd77(23)
      acd77(31)=-acd77(6)*acd77(19)
      acd77(30)=acd77(31)+acd77(27)+acd77(30)
      acd77(30)=acd77(26)*acd77(30)
      acd77(31)=acd77(1)*acd77(2)
      acd77(32)=-acd77(18)*acd77(23)
      acd77(32)=acd77(24)+acd77(32)
      acd77(32)=acd77(22)*acd77(32)
      acd77(33)=-acd77(18)*acd77(19)
      acd77(33)=acd77(20)+acd77(33)
      acd77(33)=acd77(17)*acd77(33)
      brack=acd77(28)+acd77(29)+acd77(30)-2.0_ki*acd77(31)+acd77(32)+acd77(33)
   end function brack_2
!---#] function brack_2:
!---#[ function brack_3:
   pure function brack_3(Q, mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki), dimension(29) :: acd77
      complex(ki) :: brack
      acd77(1)=d(iv1,iv2)
      acd77(2)=abb77(13)
      acd77(3)=spvak1k2(iv1)
      acd77(4)=spval3k2(iv2)
      acd77(5)=abb77(10)
      acd77(6)=spval4k2(iv2)
      acd77(7)=abb77(21)
      acd77(8)=spval4l3(iv2)
      acd77(9)=abb77(20)
      acd77(10)=spval5k2(iv2)
      acd77(11)=abb77(18)
      acd77(12)=spval5l3(iv2)
      acd77(13)=abb77(15)
      acd77(14)=spvak1k2(iv2)
      acd77(15)=spval3k2(iv1)
      acd77(16)=spval4k2(iv1)
      acd77(17)=spval4l3(iv1)
      acd77(18)=spval5k2(iv1)
      acd77(19)=spval5l3(iv1)
      acd77(20)=spvak1l3(iv2)
      acd77(21)=abb77(16)
      acd77(22)=spvak1l3(iv1)
      acd77(23)=abb77(26)
      acd77(24)=acd77(7)*acd77(6)
      acd77(25)=acd77(11)*acd77(10)
      acd77(26)=acd77(4)*acd77(5)
      acd77(27)=acd77(8)*acd77(9)
      acd77(28)=acd77(12)*acd77(13)
      acd77(24)=acd77(28)+acd77(27)+acd77(26)+acd77(24)+acd77(25)
      acd77(24)=acd77(3)*acd77(24)
      acd77(25)=acd77(16)*acd77(7)
      acd77(26)=acd77(18)*acd77(11)
      acd77(27)=acd77(15)*acd77(5)
      acd77(28)=acd77(17)*acd77(9)
      acd77(29)=acd77(19)*acd77(13)
      acd77(25)=acd77(29)+acd77(28)+acd77(27)+acd77(26)+acd77(25)
      acd77(25)=acd77(14)*acd77(25)
      acd77(26)=acd77(20)*acd77(16)
      acd77(27)=acd77(22)*acd77(6)
      acd77(26)=acd77(27)+acd77(26)
      acd77(26)=acd77(21)*acd77(26)
      acd77(27)=acd77(20)*acd77(18)
      acd77(28)=acd77(22)*acd77(10)
      acd77(27)=acd77(27)+acd77(28)
      acd77(27)=acd77(23)*acd77(27)
      acd77(28)=acd77(2)*acd77(1)
      brack=acd77(24)+acd77(25)+acd77(26)+acd77(27)+2.0_ki*acd77(28)
   end function brack_3
!---#] function brack_3:
!---#[ function derivative:
   function derivative(mu2,i1,i2) result(numerator)
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd77h1_qp
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
end module     p0_ubaru_httbar_d77h1l1d_qp
