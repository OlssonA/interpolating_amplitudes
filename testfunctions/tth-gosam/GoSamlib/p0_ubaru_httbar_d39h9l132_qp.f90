module     p0_ubaru_httbar_d39h9l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d39h9l132_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2x0mu0 = 0
   integer, parameter :: ninjaidxt1x0mu0 = 1
   integer, parameter :: ninjaidxt1x1mu0 = 2
   integer, parameter :: ninjaidxt0x0mu0 = 3
   integer, parameter :: ninjaidxt0x0mu2 = 4
   integer, parameter :: ninjaidxt0x1mu0 = 5
   integer, parameter :: ninjaidxt0x2mu0 = 6
   public :: numerator_t2
contains
!---#[ subroutine brack_21:
   pure subroutine brack_21(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(50) :: acd39
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd39(1)=dotproduct(k1,ninjaE3)
      acd39(2)=abb39(26)
      acd39(3)=dotproduct(k2,ninjaA0)
      acd39(4)=dotproduct(ninjaE3,spvak1k2)
      acd39(5)=abb39(12)
      acd39(6)=dotproduct(k2,ninjaE3)
      acd39(7)=dotproduct(ninjaA0,spvak1k2)
      acd39(8)=abb39(59)
      acd39(9)=dotproduct(ninjaA0,ninjaE3)
      acd39(10)=abb39(25)
      acd39(11)=dotproduct(ninjaE3,spval3l5)
      acd39(12)=abb39(11)
      acd39(13)=dotproduct(ninjaE3,spval4l5)
      acd39(14)=abb39(13)
      acd39(15)=dotproduct(ninjaE3,spval4l3)
      acd39(16)=abb39(14)
      acd39(17)=dotproduct(ninjaE3,spval3k2)
      acd39(18)=abb39(15)
      acd39(19)=dotproduct(ninjaE3,spvak2l3)
      acd39(20)=abb39(16)
      acd39(21)=dotproduct(ninjaA0,spval3l5)
      acd39(22)=dotproduct(ninjaA0,spval4l5)
      acd39(23)=dotproduct(ninjaA0,spval4l3)
      acd39(24)=dotproduct(ninjaA0,spval3k2)
      acd39(25)=dotproduct(ninjaA0,spvak2l3)
      acd39(26)=abb39(10)
      acd39(27)=abb39(45)
      acd39(28)=dotproduct(ninjaE3,spvak1l3)
      acd39(29)=abb39(17)
      acd39(30)=dotproduct(ninjaE3,spvak1l5)
      acd39(31)=abb39(29)
      acd39(32)=dotproduct(ninjaE3,spval4k2)
      acd39(33)=abb39(40)
      acd39(34)=dotproduct(k2,ninjaA1)
      acd39(35)=dotproduct(ninjaA1,spvak1k2)
      acd39(36)=dotproduct(ninjaA1,spval3l5)
      acd39(37)=dotproduct(ninjaA1,spval4l5)
      acd39(38)=dotproduct(ninjaA1,spval4l3)
      acd39(39)=dotproduct(ninjaA1,spval3k2)
      acd39(40)=dotproduct(ninjaA1,spvak2l3)
      acd39(41)=acd39(20)*acd39(19)
      acd39(42)=acd39(18)*acd39(17)
      acd39(43)=acd39(16)*acd39(15)
      acd39(44)=acd39(14)*acd39(13)
      acd39(45)=acd39(12)*acd39(11)
      acd39(46)=acd39(5)*acd39(6)
      acd39(41)=acd39(41)+acd39(42)+acd39(43)+acd39(44)+acd39(45)+acd39(46)
      acd39(42)=acd39(7)*acd39(41)
      acd39(43)=acd39(20)*acd39(25)
      acd39(44)=acd39(18)*acd39(24)
      acd39(45)=acd39(16)*acd39(23)
      acd39(46)=acd39(14)*acd39(22)
      acd39(47)=acd39(12)*acd39(21)
      acd39(48)=acd39(5)*acd39(3)
      acd39(43)=acd39(48)+acd39(47)+acd39(46)+acd39(45)+acd39(44)+acd39(26)+acd&
      &39(43)
      acd39(43)=acd39(4)*acd39(43)
      acd39(44)=acd39(32)*acd39(33)
      acd39(45)=acd39(30)*acd39(31)
      acd39(46)=acd39(28)*acd39(29)
      acd39(47)=acd39(9)*acd39(10)
      acd39(48)=acd39(1)*acd39(2)
      acd39(49)=acd39(17)*acd39(27)
      acd39(50)=acd39(6)*acd39(8)
      acd39(42)=acd39(43)+acd39(42)+acd39(50)+acd39(49)+acd39(48)+2.0_ki*acd39(&
      &47)+acd39(46)+acd39(44)+acd39(45)
      acd39(43)=acd39(35)*acd39(41)
      acd39(44)=acd39(20)*acd39(40)
      acd39(45)=acd39(18)*acd39(39)
      acd39(46)=acd39(16)*acd39(38)
      acd39(47)=acd39(14)*acd39(37)
      acd39(48)=acd39(12)*acd39(36)
      acd39(49)=acd39(5)*acd39(34)
      acd39(44)=acd39(49)+acd39(48)+acd39(47)+acd39(46)+acd39(44)+acd39(45)
      acd39(44)=acd39(4)*acd39(44)
      acd39(43)=acd39(43)+acd39(44)
      acd39(41)=acd39(4)*acd39(41)
      brack(ninjaidxt2x0mu0)=acd39(41)
      brack(ninjaidxt1x0mu0)=acd39(42)
      brack(ninjaidxt1x1mu0)=acd39(43)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd39h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(67) :: acd39
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd39(1)=dotproduct(k2,ninjaE3)
      acd39(2)=dotproduct(ninjaE4,spvak1k2)
      acd39(3)=abb39(12)
      acd39(4)=dotproduct(k2,ninjaE4)
      acd39(5)=dotproduct(ninjaE3,spvak1k2)
      acd39(6)=dotproduct(ninjaE4,spval3l5)
      acd39(7)=abb39(11)
      acd39(8)=dotproduct(ninjaE4,spval4l5)
      acd39(9)=abb39(13)
      acd39(10)=dotproduct(ninjaE4,spval4l3)
      acd39(11)=abb39(14)
      acd39(12)=dotproduct(ninjaE4,spval3k2)
      acd39(13)=abb39(15)
      acd39(14)=dotproduct(ninjaE4,spvak2l3)
      acd39(15)=abb39(16)
      acd39(16)=dotproduct(ninjaE3,spval3l5)
      acd39(17)=dotproduct(ninjaE3,spval4l5)
      acd39(18)=dotproduct(ninjaE3,spval4l3)
      acd39(19)=dotproduct(ninjaE3,spval3k2)
      acd39(20)=dotproduct(ninjaE3,spvak2l3)
      acd39(21)=abb39(25)
      acd39(22)=dotproduct(k1,ninjaA1)
      acd39(23)=abb39(26)
      acd39(24)=dotproduct(k2,ninjaA0)
      acd39(25)=dotproduct(ninjaA1,spvak1k2)
      acd39(26)=dotproduct(k2,ninjaA1)
      acd39(27)=dotproduct(ninjaA0,spvak1k2)
      acd39(28)=abb39(59)
      acd39(29)=dotproduct(ninjaA0,ninjaA1)
      acd39(30)=dotproduct(ninjaA1,spval3l5)
      acd39(31)=dotproduct(ninjaA1,spval4l5)
      acd39(32)=dotproduct(ninjaA1,spval4l3)
      acd39(33)=dotproduct(ninjaA1,spval3k2)
      acd39(34)=dotproduct(ninjaA1,spvak2l3)
      acd39(35)=dotproduct(ninjaA0,spval3l5)
      acd39(36)=dotproduct(ninjaA0,spval4l5)
      acd39(37)=dotproduct(ninjaA0,spval4l3)
      acd39(38)=dotproduct(ninjaA0,spval3k2)
      acd39(39)=dotproduct(ninjaA0,spvak2l3)
      acd39(40)=abb39(10)
      acd39(41)=abb39(45)
      acd39(42)=dotproduct(ninjaA1,spvak1l3)
      acd39(43)=abb39(17)
      acd39(44)=dotproduct(ninjaA1,spvak1l5)
      acd39(45)=abb39(29)
      acd39(46)=dotproduct(ninjaA1,spval4k2)
      acd39(47)=abb39(40)
      acd39(48)=dotproduct(ninjaA1,ninjaA1)
      acd39(49)=dotproduct(k1,ninjaA0)
      acd39(50)=dotproduct(ninjaA0,ninjaA0)
      acd39(51)=dotproduct(ninjaA0,spvak1l3)
      acd39(52)=dotproduct(ninjaA0,spvak1l5)
      acd39(53)=dotproduct(ninjaA0,spval4k2)
      acd39(54)=abb39(19)
      acd39(55)=acd39(15)*acd39(20)
      acd39(56)=acd39(13)*acd39(19)
      acd39(57)=acd39(11)*acd39(18)
      acd39(58)=acd39(9)*acd39(17)
      acd39(59)=acd39(7)*acd39(16)
      acd39(60)=acd39(3)*acd39(1)
      acd39(55)=acd39(60)+acd39(59)+acd39(58)+acd39(55)+acd39(56)+acd39(57)
      acd39(55)=acd39(55)*acd39(2)
      acd39(56)=acd39(15)*acd39(14)
      acd39(57)=acd39(13)*acd39(12)
      acd39(58)=acd39(11)*acd39(10)
      acd39(59)=acd39(9)*acd39(8)
      acd39(60)=acd39(7)*acd39(6)
      acd39(61)=acd39(3)*acd39(4)
      acd39(56)=acd39(61)+acd39(60)+acd39(59)+acd39(56)+acd39(57)+acd39(58)
      acd39(56)=acd39(56)*acd39(5)
      acd39(55)=acd39(21)+acd39(55)+acd39(56)
      acd39(56)=ninjaP1*acd39(55)
      acd39(57)=acd39(15)*acd39(34)
      acd39(58)=acd39(13)*acd39(33)
      acd39(59)=acd39(11)*acd39(32)
      acd39(60)=acd39(9)*acd39(31)
      acd39(61)=acd39(7)*acd39(30)
      acd39(62)=acd39(3)*acd39(26)
      acd39(57)=acd39(62)+acd39(61)+acd39(60)+acd39(59)+acd39(57)+acd39(58)
      acd39(58)=acd39(27)*acd39(57)
      acd39(59)=acd39(15)*acd39(39)
      acd39(60)=acd39(13)*acd39(38)
      acd39(61)=acd39(11)*acd39(37)
      acd39(62)=acd39(9)*acd39(36)
      acd39(63)=acd39(7)*acd39(35)
      acd39(64)=acd39(3)*acd39(24)
      acd39(59)=acd39(63)+acd39(62)+acd39(61)+acd39(59)+acd39(60)+acd39(64)+acd&
      &39(40)
      acd39(60)=acd39(25)*acd39(59)
      acd39(61)=acd39(47)*acd39(46)
      acd39(62)=acd39(45)*acd39(44)
      acd39(63)=acd39(43)*acd39(42)
      acd39(64)=acd39(23)*acd39(22)
      acd39(65)=acd39(33)*acd39(41)
      acd39(66)=acd39(26)*acd39(28)
      acd39(67)=acd39(21)*acd39(29)
      acd39(56)=acd39(60)+acd39(58)+2.0_ki*acd39(67)+acd39(66)+acd39(65)+acd39(&
      &64)+acd39(63)+acd39(61)+acd39(62)+acd39(56)
      acd39(58)=ninjaP2*acd39(55)
      acd39(57)=acd39(25)*acd39(57)
      acd39(60)=acd39(21)*acd39(48)
      acd39(57)=acd39(57)+acd39(60)+acd39(58)
      acd39(58)=ninjaP0*acd39(55)
      acd39(59)=acd39(27)*acd39(59)
      acd39(60)=acd39(47)*acd39(53)
      acd39(61)=acd39(45)*acd39(52)
      acd39(62)=acd39(43)*acd39(51)
      acd39(63)=acd39(23)*acd39(49)
      acd39(64)=acd39(38)*acd39(41)
      acd39(65)=acd39(24)*acd39(28)
      acd39(66)=acd39(21)*acd39(50)
      acd39(58)=acd39(59)+acd39(66)+acd39(65)+acd39(64)+acd39(63)+acd39(62)+acd&
      &39(61)+acd39(54)+acd39(60)+acd39(58)
      brack(ninjaidxt0x0mu0)=acd39(58)
      brack(ninjaidxt0x0mu2)=acd39(55)
      brack(ninjaidxt0x1mu0)=acd39(56)
      brack(ninjaidxt0x2mu0)=acd39(57)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d39h9_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd39h9_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k3-k4-k5
      vecA0(1:4) = - a0(0:3) - qshift(1:4)
      vecA1(1:4) = - a1(0:3)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_ubaru_httbar_d39h9l132_qp
