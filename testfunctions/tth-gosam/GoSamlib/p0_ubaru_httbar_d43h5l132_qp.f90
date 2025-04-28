module     p0_ubaru_httbar_d43h5l132_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity5d43h5l132_qp.f90
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
      use p0_ubaru_httbar_abbrevd43h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(47) :: acd43
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd43(1)=dotproduct(k2,ninjaA0)
      acd43(2)=dotproduct(ninjaE3,spvak1k2)
      acd43(3)=abb43(12)
      acd43(4)=dotproduct(k2,ninjaE3)
      acd43(5)=dotproduct(ninjaA0,spvak1k2)
      acd43(6)=abb43(19)
      acd43(7)=dotproduct(ninjaA0,ninjaE3)
      acd43(8)=abb43(21)
      acd43(9)=dotproduct(ninjaE3,spval5l4)
      acd43(10)=abb43(13)
      acd43(11)=dotproduct(ninjaE3,spval5l3)
      acd43(12)=abb43(14)
      acd43(13)=dotproduct(ninjaE3,spval3k2)
      acd43(14)=abb43(15)
      acd43(15)=dotproduct(ninjaE3,spval3l4)
      acd43(16)=abb43(16)
      acd43(17)=dotproduct(ninjaE3,spvak2l3)
      acd43(18)=abb43(18)
      acd43(19)=dotproduct(ninjaA0,spval5l4)
      acd43(20)=dotproduct(ninjaA0,spval5l3)
      acd43(21)=dotproduct(ninjaA0,spval3k2)
      acd43(22)=dotproduct(ninjaA0,spval3l4)
      acd43(23)=dotproduct(ninjaA0,spvak2l3)
      acd43(24)=dotproduct(ninjaE3,spval5k2)
      acd43(25)=abb43(10)
      acd43(26)=abb43(20)
      acd43(27)=abb43(28)
      acd43(28)=dotproduct(ninjaE3,spvak1l4)
      acd43(29)=abb43(17)
      acd43(30)=dotproduct(ninjaE3,spvak1l3)
      acd43(31)=abb43(24)
      acd43(32)=dotproduct(k2,ninjaA1)
      acd43(33)=dotproduct(ninjaA1,spvak1k2)
      acd43(34)=dotproduct(ninjaA1,spval5l4)
      acd43(35)=dotproduct(ninjaA1,spval5l3)
      acd43(36)=dotproduct(ninjaA1,spval3k2)
      acd43(37)=dotproduct(ninjaA1,spval3l4)
      acd43(38)=dotproduct(ninjaA1,spvak2l3)
      acd43(39)=acd43(18)*acd43(17)
      acd43(40)=acd43(16)*acd43(15)
      acd43(41)=acd43(14)*acd43(13)
      acd43(42)=acd43(12)*acd43(11)
      acd43(43)=acd43(10)*acd43(9)
      acd43(44)=acd43(3)*acd43(4)
      acd43(39)=acd43(39)+acd43(40)+acd43(41)+acd43(42)+acd43(43)+acd43(44)
      acd43(40)=acd43(5)*acd43(39)
      acd43(41)=acd43(18)*acd43(23)
      acd43(42)=acd43(16)*acd43(22)
      acd43(43)=acd43(14)*acd43(21)
      acd43(44)=acd43(12)*acd43(20)
      acd43(45)=acd43(10)*acd43(19)
      acd43(46)=acd43(3)*acd43(1)
      acd43(41)=acd43(46)+acd43(45)+acd43(44)+acd43(43)+acd43(42)+acd43(26)+acd&
      &43(41)
      acd43(41)=acd43(2)*acd43(41)
      acd43(42)=acd43(30)*acd43(31)
      acd43(43)=acd43(28)*acd43(29)
      acd43(44)=acd43(24)*acd43(25)
      acd43(45)=acd43(7)*acd43(8)
      acd43(46)=acd43(13)*acd43(27)
      acd43(47)=acd43(4)*acd43(6)
      acd43(40)=acd43(41)+acd43(40)+acd43(47)+acd43(46)+2.0_ki*acd43(45)+acd43(&
      &44)+acd43(42)+acd43(43)
      acd43(41)=acd43(33)*acd43(39)
      acd43(42)=acd43(18)*acd43(38)
      acd43(43)=acd43(16)*acd43(37)
      acd43(44)=acd43(14)*acd43(36)
      acd43(45)=acd43(12)*acd43(35)
      acd43(46)=acd43(10)*acd43(34)
      acd43(47)=acd43(3)*acd43(32)
      acd43(42)=acd43(47)+acd43(46)+acd43(45)+acd43(44)+acd43(42)+acd43(43)
      acd43(42)=acd43(2)*acd43(42)
      acd43(41)=acd43(41)+acd43(42)
      acd43(39)=acd43(2)*acd43(39)
      brack(ninjaidxt2x0mu0)=acd43(39)
      brack(ninjaidxt1x0mu0)=acd43(40)
      brack(ninjaidxt1x1mu0)=acd43(41)
   end subroutine brack_21
!---#] subroutine brack_21:
!---#[ subroutine brack_22:
   pure subroutine brack_22(ninjaA0, ninjaA1, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd43h5_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA0, ninjaA1, ninjaE3, ninja&
      &E4
      complex(ki), dimension(0:2), intent(in) :: ninjaP
      complex(ki), dimension(63) :: acd43
      complex(ki), dimension (0:*), intent(inout) :: brack
      complex(ki) :: ninjaP0, ninjaP1, ninjaP2
      ninjaP0 = ninjaP(0)
      ninjaP1 = ninjaP(1)
      ninjaP2 = ninjaP(2)
      acd43(1)=dotproduct(k2,ninjaE3)
      acd43(2)=dotproduct(ninjaE4,spvak1k2)
      acd43(3)=abb43(12)
      acd43(4)=dotproduct(k2,ninjaE4)
      acd43(5)=dotproduct(ninjaE3,spvak1k2)
      acd43(6)=dotproduct(ninjaE4,spval5l4)
      acd43(7)=abb43(13)
      acd43(8)=dotproduct(ninjaE4,spval5l3)
      acd43(9)=abb43(14)
      acd43(10)=dotproduct(ninjaE4,spval3k2)
      acd43(11)=abb43(15)
      acd43(12)=dotproduct(ninjaE4,spval3l4)
      acd43(13)=abb43(16)
      acd43(14)=dotproduct(ninjaE4,spvak2l3)
      acd43(15)=abb43(18)
      acd43(16)=dotproduct(ninjaE3,spval5l4)
      acd43(17)=dotproduct(ninjaE3,spval5l3)
      acd43(18)=dotproduct(ninjaE3,spval3k2)
      acd43(19)=dotproduct(ninjaE3,spval3l4)
      acd43(20)=dotproduct(ninjaE3,spvak2l3)
      acd43(21)=abb43(21)
      acd43(22)=dotproduct(k2,ninjaA0)
      acd43(23)=dotproduct(ninjaA1,spvak1k2)
      acd43(24)=dotproduct(k2,ninjaA1)
      acd43(25)=dotproduct(ninjaA0,spvak1k2)
      acd43(26)=abb43(19)
      acd43(27)=dotproduct(ninjaA0,ninjaA1)
      acd43(28)=dotproduct(ninjaA1,spval5l4)
      acd43(29)=dotproduct(ninjaA1,spval5l3)
      acd43(30)=dotproduct(ninjaA1,spval3k2)
      acd43(31)=dotproduct(ninjaA1,spval3l4)
      acd43(32)=dotproduct(ninjaA1,spvak2l3)
      acd43(33)=dotproduct(ninjaA0,spval5l4)
      acd43(34)=dotproduct(ninjaA0,spval5l3)
      acd43(35)=dotproduct(ninjaA0,spval3k2)
      acd43(36)=dotproduct(ninjaA0,spval3l4)
      acd43(37)=dotproduct(ninjaA0,spvak2l3)
      acd43(38)=dotproduct(ninjaA1,spval5k2)
      acd43(39)=abb43(10)
      acd43(40)=abb43(20)
      acd43(41)=abb43(28)
      acd43(42)=dotproduct(ninjaA1,spvak1l4)
      acd43(43)=abb43(17)
      acd43(44)=dotproduct(ninjaA1,spvak1l3)
      acd43(45)=abb43(24)
      acd43(46)=dotproduct(ninjaA1,ninjaA1)
      acd43(47)=dotproduct(ninjaA0,ninjaA0)
      acd43(48)=dotproduct(ninjaA0,spval5k2)
      acd43(49)=dotproduct(ninjaA0,spvak1l4)
      acd43(50)=dotproduct(ninjaA0,spvak1l3)
      acd43(51)=abb43(11)
      acd43(52)=acd43(15)*acd43(20)
      acd43(53)=acd43(13)*acd43(19)
      acd43(54)=acd43(11)*acd43(18)
      acd43(55)=acd43(9)*acd43(17)
      acd43(56)=acd43(7)*acd43(16)
      acd43(57)=acd43(3)*acd43(1)
      acd43(52)=acd43(57)+acd43(56)+acd43(55)+acd43(52)+acd43(53)+acd43(54)
      acd43(52)=acd43(52)*acd43(2)
      acd43(53)=acd43(15)*acd43(14)
      acd43(54)=acd43(13)*acd43(12)
      acd43(55)=acd43(11)*acd43(10)
      acd43(56)=acd43(9)*acd43(8)
      acd43(57)=acd43(7)*acd43(6)
      acd43(58)=acd43(3)*acd43(4)
      acd43(53)=acd43(58)+acd43(57)+acd43(56)+acd43(53)+acd43(54)+acd43(55)
      acd43(53)=acd43(53)*acd43(5)
      acd43(52)=acd43(21)+acd43(52)+acd43(53)
      acd43(53)=ninjaP1*acd43(52)
      acd43(54)=acd43(15)*acd43(32)
      acd43(55)=acd43(13)*acd43(31)
      acd43(56)=acd43(11)*acd43(30)
      acd43(57)=acd43(9)*acd43(29)
      acd43(58)=acd43(7)*acd43(28)
      acd43(59)=acd43(3)*acd43(24)
      acd43(54)=acd43(59)+acd43(58)+acd43(57)+acd43(56)+acd43(54)+acd43(55)
      acd43(55)=acd43(25)*acd43(54)
      acd43(56)=acd43(15)*acd43(37)
      acd43(57)=acd43(13)*acd43(36)
      acd43(58)=acd43(11)*acd43(35)
      acd43(59)=acd43(9)*acd43(34)
      acd43(60)=acd43(7)*acd43(33)
      acd43(61)=acd43(3)*acd43(22)
      acd43(56)=acd43(60)+acd43(59)+acd43(58)+acd43(56)+acd43(57)+acd43(61)+acd&
      &43(40)
      acd43(57)=acd43(23)*acd43(56)
      acd43(58)=acd43(45)*acd43(44)
      acd43(59)=acd43(43)*acd43(42)
      acd43(60)=acd43(39)*acd43(38)
      acd43(61)=acd43(30)*acd43(41)
      acd43(62)=acd43(24)*acd43(26)
      acd43(63)=acd43(21)*acd43(27)
      acd43(53)=acd43(57)+acd43(55)+2.0_ki*acd43(63)+acd43(62)+acd43(61)+acd43(&
      &60)+acd43(58)+acd43(59)+acd43(53)
      acd43(55)=ninjaP2*acd43(52)
      acd43(54)=acd43(23)*acd43(54)
      acd43(57)=acd43(21)*acd43(46)
      acd43(54)=acd43(54)+acd43(57)+acd43(55)
      acd43(55)=ninjaP0*acd43(52)
      acd43(56)=acd43(25)*acd43(56)
      acd43(57)=acd43(45)*acd43(50)
      acd43(58)=acd43(43)*acd43(49)
      acd43(59)=acd43(39)*acd43(48)
      acd43(60)=acd43(35)*acd43(41)
      acd43(61)=acd43(22)*acd43(26)
      acd43(62)=acd43(21)*acd43(47)
      acd43(55)=acd43(56)+acd43(62)+acd43(61)+acd43(60)+acd43(59)+acd43(58)+acd&
      &43(51)+acd43(57)+acd43(55)
      brack(ninjaidxt0x0mu0)=acd43(55)
      brack(ninjaidxt0x0mu2)=acd43(52)
      brack(ninjaidxt0x1mu0)=acd43(53)
      brack(ninjaidxt0x2mu0)=acd43(54)
   end subroutine brack_22
!---#] subroutine brack_22:
!---#[ subroutine numerator_t2:
   subroutine numerator_t2(ncut, a0, a1, b, c, param, deg, coeffs) &
   & bind(c, name="p0_ubaru_httbar_d43h5_qp_ninja_t2")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd43h5_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a0, a1, b, c
      complex(ki), dimension(0:2), intent(in) :: param
      complex(ki), dimension(4) :: vecA0, vecA1, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
      vecA0(1:4) = + a0(0:3) - qshift(1:4)
      vecA1(1:4) = + a1(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_21,vecA0,vecA1,vecB,vecC,param,coeffs)
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_22,vecA0,vecA1,vecB,vecC,param,coeffs)
   end subroutine numerator_t2
!---#] subroutine numerator_t2:
end module     p0_ubaru_httbar_d43h5l132_qp
