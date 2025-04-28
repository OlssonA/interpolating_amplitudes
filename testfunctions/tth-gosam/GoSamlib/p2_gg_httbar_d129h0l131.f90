module     p2_gg_httbar_d129h0l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d129h0l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt2mu0 = 0
   integer, parameter :: ninjaidxt1mu0 = 1
   integer, parameter :: ninjaidxt0mu0 = 2
   integer, parameter :: ninjaidxt0mu2 = 3
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd129h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd129
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd129h0
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(86) :: acd129
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd129(1)=abb129(22)
      acd129(2)=dotproduct(k2,ninjaE3)
      acd129(3)=abb129(25)
      acd129(4)=dotproduct(l3,ninjaE3)
      acd129(5)=abb129(23)
      acd129(6)=dotproduct(l5,ninjaE3)
      acd129(7)=abb129(21)
      acd129(8)=dotproduct(ninjaA,ninjaE3)
      acd129(9)=dotproduct(ninjaE3,spval3k2)
      acd129(10)=abb129(13)
      acd129(11)=dotproduct(ninjaE3,spval5k2)
      acd129(12)=abb129(14)
      acd129(13)=dotproduct(ninjaE3,spval5k1)
      acd129(14)=abb129(15)
      acd129(15)=dotproduct(ninjaE3,spval3l5)
      acd129(16)=abb129(16)
      acd129(17)=dotproduct(ninjaE3,spval3k1)
      acd129(18)=abb129(17)
      acd129(19)=dotproduct(ninjaE3,spvak2l3)
      acd129(20)=abb129(19)
      acd129(21)=dotproduct(ninjaE3,spvak1l3)
      acd129(22)=abb129(20)
      acd129(23)=dotproduct(ninjaE3,spvak1k2)
      acd129(24)=abb129(24)
      acd129(25)=dotproduct(ninjaE3,spval5e1)
      acd129(26)=abb129(26)
      acd129(27)=dotproduct(ninjaE3,spvae1k2)
      acd129(28)=abb129(27)
      acd129(29)=dotproduct(ninjaE3,spval5e2)
      acd129(30)=abb129(40)
      acd129(31)=dotproduct(ninjaE3,spvae2k2)
      acd129(32)=abb129(44)
      acd129(33)=dotproduct(ninjaE3,spval5l3)
      acd129(34)=abb129(48)
      acd129(35)=dotproduct(ninjaE3,spval3e2)
      acd129(36)=abb129(59)
      acd129(37)=dotproduct(ninjaE3,spvae2l3)
      acd129(38)=abb129(67)
      acd129(39)=dotproduct(ninjaE3,spval3e1)
      acd129(40)=abb129(68)
      acd129(41)=dotproduct(ninjaE3,spvae1l3)
      acd129(42)=abb129(179)
      acd129(43)=dotproduct(k2,ninjaA)
      acd129(44)=dotproduct(l3,ninjaA)
      acd129(45)=dotproduct(l5,ninjaA)
      acd129(46)=dotproduct(ninjaA,ninjaA)
      acd129(47)=dotproduct(ninjaA,spval3k2)
      acd129(48)=dotproduct(ninjaA,spval5k2)
      acd129(49)=dotproduct(ninjaA,spval5k1)
      acd129(50)=dotproduct(ninjaA,spval3l5)
      acd129(51)=dotproduct(ninjaA,spval3k1)
      acd129(52)=dotproduct(ninjaA,spvak2l3)
      acd129(53)=dotproduct(ninjaA,spvak1l3)
      acd129(54)=dotproduct(ninjaA,spvak1k2)
      acd129(55)=dotproduct(ninjaA,spval5e1)
      acd129(56)=dotproduct(ninjaA,spvae1k2)
      acd129(57)=dotproduct(ninjaA,spval5e2)
      acd129(58)=dotproduct(ninjaA,spvae2k2)
      acd129(59)=dotproduct(ninjaA,spval5l3)
      acd129(60)=dotproduct(ninjaA,spval3e2)
      acd129(61)=dotproduct(ninjaA,spvae2l3)
      acd129(62)=dotproduct(ninjaA,spval3e1)
      acd129(63)=dotproduct(ninjaA,spvae1l3)
      acd129(64)=abb129(18)
      acd129(65)=acd129(2)*acd129(3)
      acd129(66)=acd129(4)*acd129(5)
      acd129(67)=acd129(6)*acd129(7)
      acd129(68)=acd129(8)*acd129(1)
      acd129(69)=acd129(9)*acd129(10)
      acd129(70)=acd129(11)*acd129(12)
      acd129(71)=acd129(13)*acd129(14)
      acd129(72)=acd129(15)*acd129(16)
      acd129(73)=acd129(17)*acd129(18)
      acd129(74)=acd129(19)*acd129(20)
      acd129(75)=acd129(21)*acd129(22)
      acd129(76)=acd129(23)*acd129(24)
      acd129(77)=acd129(25)*acd129(26)
      acd129(78)=acd129(27)*acd129(28)
      acd129(79)=acd129(29)*acd129(30)
      acd129(80)=acd129(31)*acd129(32)
      acd129(81)=acd129(33)*acd129(34)
      acd129(82)=acd129(35)*acd129(36)
      acd129(83)=acd129(37)*acd129(38)
      acd129(84)=acd129(39)*acd129(40)
      acd129(85)=-acd129(41)*acd129(42)
      acd129(65)=acd129(85)+acd129(84)+acd129(83)+acd129(82)+acd129(81)+acd129(&
      &80)+acd129(79)+acd129(78)+acd129(77)+acd129(76)+acd129(75)+acd129(74)+ac&
      &d129(73)+acd129(72)+acd129(71)+acd129(70)+acd129(69)+2.0_ki*acd129(68)+a&
      &cd129(67)+acd129(65)+acd129(66)
      acd129(66)=ninjaP+acd129(46)
      acd129(66)=acd129(1)*acd129(66)
      acd129(67)=acd129(43)*acd129(3)
      acd129(68)=acd129(44)*acd129(5)
      acd129(69)=acd129(45)*acd129(7)
      acd129(70)=acd129(47)*acd129(10)
      acd129(71)=acd129(48)*acd129(12)
      acd129(72)=acd129(49)*acd129(14)
      acd129(73)=acd129(50)*acd129(16)
      acd129(74)=acd129(51)*acd129(18)
      acd129(75)=acd129(52)*acd129(20)
      acd129(76)=acd129(53)*acd129(22)
      acd129(77)=acd129(54)*acd129(24)
      acd129(78)=acd129(55)*acd129(26)
      acd129(79)=acd129(56)*acd129(28)
      acd129(80)=acd129(57)*acd129(30)
      acd129(81)=acd129(58)*acd129(32)
      acd129(82)=acd129(59)*acd129(34)
      acd129(83)=acd129(60)*acd129(36)
      acd129(84)=acd129(61)*acd129(38)
      acd129(85)=acd129(62)*acd129(40)
      acd129(86)=-acd129(63)*acd129(42)
      acd129(66)=acd129(64)+acd129(86)+acd129(85)+acd129(84)+acd129(83)+acd129(&
      &82)+acd129(81)+acd129(80)+acd129(79)+acd129(78)+acd129(77)+acd129(76)+ac&
      &d129(75)+acd129(74)+acd129(73)+acd129(72)+acd129(71)+acd129(70)+acd129(6&
      &9)+acd129(67)+acd129(68)+acd129(66)
      brack(ninjaidxt1mu0)=acd129(65)
      brack(ninjaidxt0mu0)=acd129(66)
      brack(ninjaidxt0mu2)=acd129(1)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d129h0_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd129h0
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k5
      vecA(1:4) = + a(0:3) - qshift(1:4)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d129h0l131
