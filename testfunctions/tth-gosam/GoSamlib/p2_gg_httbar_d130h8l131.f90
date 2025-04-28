module     p2_gg_httbar_d130h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d130h8l131.f90
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
      use p2_gg_httbar_abbrevd130h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd130
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd130h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(86) :: acd130
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd130(1)=abb130(26)
      acd130(2)=dotproduct(k2,ninjaE3)
      acd130(3)=abb130(21)
      acd130(4)=dotproduct(l3,ninjaE3)
      acd130(5)=abb130(86)
      acd130(6)=dotproduct(l4,ninjaE3)
      acd130(7)=abb130(39)
      acd130(8)=dotproduct(ninjaA,ninjaE3)
      acd130(9)=dotproduct(ninjaE3,spval4k2)
      acd130(10)=abb130(13)
      acd130(11)=dotproduct(ninjaE3,spvae1k2)
      acd130(12)=abb130(14)
      acd130(13)=dotproduct(ninjaE3,spvak2l3)
      acd130(14)=abb130(15)
      acd130(15)=dotproduct(ninjaE3,spval4l3)
      acd130(16)=abb130(16)
      acd130(17)=dotproduct(ninjaE3,spvak1k2)
      acd130(18)=abb130(17)
      acd130(19)=dotproduct(ninjaE3,spval4k1)
      acd130(20)=abb130(18)
      acd130(21)=dotproduct(ninjaE3,spval4e1)
      acd130(22)=abb130(20)
      acd130(23)=dotproduct(ninjaE3,spval3l4)
      acd130(24)=abb130(22)
      acd130(25)=dotproduct(ninjaE3,spval3k2)
      acd130(26)=abb130(23)
      acd130(27)=dotproduct(ninjaE3,spvae2k2)
      acd130(28)=abb130(27)
      acd130(29)=dotproduct(ninjaE3,spval4e2)
      acd130(30)=abb130(29)
      acd130(31)=dotproduct(ninjaE3,spvae2l3)
      acd130(32)=abb130(34)
      acd130(33)=dotproduct(ninjaE3,spvae1l3)
      acd130(34)=abb130(36)
      acd130(35)=dotproduct(ninjaE3,spval3e2)
      acd130(36)=abb130(40)
      acd130(37)=dotproduct(ninjaE3,spval3e1)
      acd130(38)=abb130(41)
      acd130(39)=dotproduct(ninjaE3,spvak1l3)
      acd130(40)=abb130(161)
      acd130(41)=dotproduct(ninjaE3,spval3k1)
      acd130(42)=abb130(179)
      acd130(43)=dotproduct(k2,ninjaA)
      acd130(44)=dotproduct(l3,ninjaA)
      acd130(45)=dotproduct(l4,ninjaA)
      acd130(46)=dotproduct(ninjaA,ninjaA)
      acd130(47)=dotproduct(ninjaA,spval4k2)
      acd130(48)=dotproduct(ninjaA,spvae1k2)
      acd130(49)=dotproduct(ninjaA,spvak2l3)
      acd130(50)=dotproduct(ninjaA,spval4l3)
      acd130(51)=dotproduct(ninjaA,spvak1k2)
      acd130(52)=dotproduct(ninjaA,spval4k1)
      acd130(53)=dotproduct(ninjaA,spval4e1)
      acd130(54)=dotproduct(ninjaA,spval3l4)
      acd130(55)=dotproduct(ninjaA,spval3k2)
      acd130(56)=dotproduct(ninjaA,spvae2k2)
      acd130(57)=dotproduct(ninjaA,spval4e2)
      acd130(58)=dotproduct(ninjaA,spvae2l3)
      acd130(59)=dotproduct(ninjaA,spvae1l3)
      acd130(60)=dotproduct(ninjaA,spval3e2)
      acd130(61)=dotproduct(ninjaA,spval3e1)
      acd130(62)=dotproduct(ninjaA,spvak1l3)
      acd130(63)=dotproduct(ninjaA,spval3k1)
      acd130(64)=abb130(19)
      acd130(65)=acd130(2)*acd130(3)
      acd130(66)=acd130(4)*acd130(5)
      acd130(67)=acd130(6)*acd130(7)
      acd130(68)=acd130(8)*acd130(1)
      acd130(69)=acd130(9)*acd130(10)
      acd130(70)=acd130(11)*acd130(12)
      acd130(71)=acd130(13)*acd130(14)
      acd130(72)=acd130(15)*acd130(16)
      acd130(73)=acd130(17)*acd130(18)
      acd130(74)=acd130(19)*acd130(20)
      acd130(75)=acd130(21)*acd130(22)
      acd130(76)=acd130(23)*acd130(24)
      acd130(77)=acd130(25)*acd130(26)
      acd130(78)=acd130(27)*acd130(28)
      acd130(79)=acd130(29)*acd130(30)
      acd130(80)=acd130(31)*acd130(32)
      acd130(81)=acd130(33)*acd130(34)
      acd130(82)=acd130(35)*acd130(36)
      acd130(83)=acd130(37)*acd130(38)
      acd130(84)=-acd130(39)*acd130(40)
      acd130(85)=-acd130(41)*acd130(42)
      acd130(65)=acd130(85)+acd130(84)+acd130(83)+acd130(82)+acd130(81)+acd130(&
      &80)+acd130(79)+acd130(78)+acd130(77)+acd130(76)+acd130(75)+acd130(74)+ac&
      &d130(73)+acd130(72)+acd130(71)+acd130(70)+acd130(69)+2.0_ki*acd130(68)+a&
      &cd130(67)+acd130(65)+acd130(66)
      acd130(66)=ninjaP+acd130(46)
      acd130(66)=acd130(1)*acd130(66)
      acd130(67)=acd130(43)*acd130(3)
      acd130(68)=acd130(44)*acd130(5)
      acd130(69)=acd130(45)*acd130(7)
      acd130(70)=acd130(47)*acd130(10)
      acd130(71)=acd130(48)*acd130(12)
      acd130(72)=acd130(49)*acd130(14)
      acd130(73)=acd130(50)*acd130(16)
      acd130(74)=acd130(51)*acd130(18)
      acd130(75)=acd130(52)*acd130(20)
      acd130(76)=acd130(53)*acd130(22)
      acd130(77)=acd130(54)*acd130(24)
      acd130(78)=acd130(55)*acd130(26)
      acd130(79)=acd130(56)*acd130(28)
      acd130(80)=acd130(57)*acd130(30)
      acd130(81)=acd130(58)*acd130(32)
      acd130(82)=acd130(59)*acd130(34)
      acd130(83)=acd130(60)*acd130(36)
      acd130(84)=acd130(61)*acd130(38)
      acd130(85)=-acd130(62)*acd130(40)
      acd130(86)=-acd130(63)*acd130(42)
      acd130(66)=acd130(64)+acd130(86)+acd130(85)+acd130(84)+acd130(83)+acd130(&
      &82)+acd130(81)+acd130(80)+acd130(79)+acd130(78)+acd130(77)+acd130(76)+ac&
      &d130(75)+acd130(74)+acd130(73)+acd130(72)+acd130(71)+acd130(70)+acd130(6&
      &9)+acd130(67)+acd130(68)+acd130(66)
      brack(ninjaidxt1mu0)=acd130(65)
      brack(ninjaidxt0mu0)=acd130(66)
      brack(ninjaidxt0mu2)=acd130(1)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d130h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd130h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = -k4
      vecA(1:4) = - a(0:3) - qshift(1:4)
      vecB(1:4) = - b(0:3)
      vecC(1:4) = - c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
      if (deg.le.(1+(-1))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d130h8l131
