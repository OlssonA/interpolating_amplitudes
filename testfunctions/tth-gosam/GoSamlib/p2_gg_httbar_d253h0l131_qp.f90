module     p2_gg_httbar_d253h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d253h0l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond_t, d => metric_tensor
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
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd253h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd253
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd253h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(82) :: acd253
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd253(1)=dotproduct(ninjaE3,spval5e2)
      acd253(2)=dotproduct(ninjaE3,spvae2e1)
      acd253(3)=abb253(44)
      acd253(4)=dotproduct(ninjaE3,spval4e2)
      acd253(5)=abb253(59)
      acd253(6)=dotproduct(ninjaA,ninjaE3)
      acd253(7)=dotproduct(ninjaE3,spvae2k2)
      acd253(8)=dotproduct(ninjaE3,spval4e1)
      acd253(9)=dotproduct(ninjaE3,spvae1e2)
      acd253(10)=abb253(7)
      acd253(11)=dotproduct(ninjaE3,spval5e1)
      acd253(12)=abb253(38)
      acd253(13)=dotproduct(ninjaE3,spvae2k1)
      acd253(14)=abb253(8)
      acd253(15)=dotproduct(ninjaE3,spvae2l3)
      acd253(16)=abb253(56)
      acd253(17)=abb253(32)
      acd253(18)=abb253(49)
      acd253(19)=dotproduct(ninjaE3,spvae1k2)
      acd253(20)=abb253(48)
      acd253(21)=dotproduct(ninjaA,ninjaA)
      acd253(22)=dotproduct(ninjaA,spval5e2)
      acd253(23)=dotproduct(ninjaA,spvae2e1)
      acd253(24)=dotproduct(ninjaA,spval4e2)
      acd253(25)=abb253(24)
      acd253(26)=abb253(40)
      acd253(27)=abb253(46)
      acd253(28)=abb253(37)
      acd253(29)=abb253(58)
      acd253(30)=dotproduct(ninjaA,spvae2k2)
      acd253(31)=dotproduct(ninjaA,spval4e1)
      acd253(32)=dotproduct(ninjaA,spvae1e2)
      acd253(33)=dotproduct(ninjaA,spvae2k1)
      acd253(34)=dotproduct(ninjaA,spval5e1)
      acd253(35)=dotproduct(ninjaA,spvae1k2)
      acd253(36)=dotproduct(ninjaA,spvae2l3)
      acd253(37)=abb253(18)
      acd253(38)=dotproduct(ninjaE3,spvae1l3)
      acd253(39)=abb253(19)
      acd253(40)=dotproduct(ninjaE3,spvae1k1)
      acd253(41)=abb253(36)
      acd253(42)=abb253(39)
      acd253(43)=dotproduct(ninjaE3,spval4k1)
      acd253(44)=abb253(43)
      acd253(45)=abb253(17)
      acd253(46)=dotproduct(ninjaE3,spval5k2)
      acd253(47)=abb253(11)
      acd253(48)=dotproduct(ninjaE3,spvak1k2)
      acd253(49)=abb253(27)
      acd253(50)=dotproduct(ninjaE3,spval3e1)
      acd253(51)=abb253(50)
      acd253(52)=abb253(55)
      acd253(53)=dotproduct(ninjaE3,spval4k2)
      acd253(54)=abb253(34)
      acd253(55)=abb253(41)
      acd253(56)=dotproduct(ninjaE3,spvak1e1)
      acd253(57)=abb253(53)
      acd253(58)=abb253(9)
      acd253(59)=abb253(14)
      acd253(60)=abb253(23)
      acd253(61)=abb253(54)
      acd253(62)=abb253(60)
      acd253(63)=abb253(31)
      acd253(64)=abb253(45)
      acd253(65)=abb253(47)
      acd253(66)=abb253(51)
      acd253(67)=acd253(4)*acd253(5)
      acd253(68)=acd253(1)*acd253(3)
      acd253(68)=acd253(67)+acd253(68)
      acd253(68)=acd253(2)*acd253(68)
      acd253(69)=acd253(15)*acd253(18)
      acd253(70)=acd253(13)*acd253(17)
      acd253(69)=acd253(70)+acd253(69)
      acd253(69)=acd253(11)*acd253(69)
      acd253(70)=acd253(13)*acd253(14)
      acd253(71)=acd253(15)*acd253(16)
      acd253(70)=acd253(70)+acd253(71)
      acd253(70)=acd253(8)*acd253(70)
      acd253(71)=acd253(11)*acd253(12)
      acd253(72)=acd253(8)*acd253(10)
      acd253(71)=acd253(71)+acd253(72)
      acd253(72)=acd253(7)*acd253(71)
      acd253(69)=acd253(72)+acd253(69)+acd253(70)
      acd253(69)=acd253(9)*acd253(69)
      acd253(70)=2.0_ki*acd253(6)
      acd253(67)=acd253(70)*acd253(67)
      acd253(72)=acd253(19)*acd253(20)
      acd253(73)=acd253(3)*acd253(70)
      acd253(73)=acd253(72)+acd253(73)
      acd253(73)=acd253(1)*acd253(73)
      acd253(67)=acd253(67)+acd253(73)
      acd253(67)=acd253(2)*acd253(67)
      acd253(67)=acd253(67)+acd253(69)
      acd253(69)=acd253(18)*acd253(36)
      acd253(73)=acd253(17)*acd253(33)
      acd253(74)=acd253(12)*acd253(30)
      acd253(69)=acd253(74)+acd253(73)+acd253(52)+acd253(69)
      acd253(69)=acd253(11)*acd253(69)
      acd253(73)=acd253(16)*acd253(36)
      acd253(74)=acd253(14)*acd253(33)
      acd253(75)=acd253(10)*acd253(30)
      acd253(73)=acd253(75)+acd253(74)+acd253(45)+acd253(73)
      acd253(73)=acd253(8)*acd253(73)
      acd253(74)=acd253(17)*acd253(34)
      acd253(75)=acd253(14)*acd253(31)
      acd253(74)=acd253(74)+acd253(75)
      acd253(74)=acd253(13)*acd253(74)
      acd253(75)=acd253(18)*acd253(34)
      acd253(76)=acd253(16)*acd253(31)
      acd253(75)=acd253(76)+acd253(55)+acd253(75)
      acd253(75)=acd253(15)*acd253(75)
      acd253(76)=acd253(12)*acd253(34)
      acd253(77)=acd253(10)*acd253(31)
      acd253(76)=acd253(77)+acd253(37)+acd253(76)
      acd253(76)=acd253(7)*acd253(76)
      acd253(77)=acd253(53)*acd253(54)
      acd253(78)=acd253(46)*acd253(47)
      acd253(79)=acd253(56)*acd253(57)
      acd253(80)=-acd253(50)*acd253(51)
      acd253(81)=acd253(48)*acd253(49)
      acd253(82)=acd253(26)*acd253(70)
      acd253(69)=acd253(76)+acd253(82)+acd253(73)+acd253(69)+acd253(75)+acd253(&
      &74)+acd253(81)+acd253(80)+acd253(79)+acd253(77)+acd253(78)
      acd253(69)=acd253(9)*acd253(69)
      acd253(73)=acd253(5)*acd253(24)
      acd253(74)=acd253(3)*acd253(22)
      acd253(73)=acd253(74)+acd253(28)+acd253(73)
      acd253(73)=acd253(73)*acd253(70)
      acd253(74)=acd253(20)*acd253(35)
      acd253(75)=acd253(21)+ninjaP
      acd253(76)=acd253(3)*acd253(75)
      acd253(74)=acd253(76)+acd253(58)+acd253(74)
      acd253(74)=acd253(1)*acd253(74)
      acd253(76)=-acd253(43)*acd253(66)
      acd253(77)=acd253(40)*acd253(64)
      acd253(78)=acd253(38)*acd253(62)
      acd253(79)=acd253(20)*acd253(22)
      acd253(79)=acd253(65)+acd253(79)
      acd253(79)=acd253(19)*acd253(79)
      acd253(75)=acd253(5)*acd253(75)
      acd253(75)=acd253(63)+acd253(75)
      acd253(75)=acd253(4)*acd253(75)
      acd253(73)=acd253(74)+acd253(73)+acd253(75)+acd253(79)+acd253(78)+acd253(&
      &76)+acd253(77)
      acd253(73)=acd253(2)*acd253(73)
      acd253(71)=acd253(32)*acd253(71)
      acd253(74)=-acd253(43)*acd253(44)
      acd253(75)=acd253(40)*acd253(41)
      acd253(76)=acd253(38)*acd253(39)
      acd253(77)=acd253(19)*acd253(42)
      acd253(78)=acd253(25)*acd253(70)
      acd253(71)=acd253(78)+acd253(77)+acd253(76)+acd253(74)+acd253(75)+acd253(&
      &71)
      acd253(71)=acd253(7)*acd253(71)
      acd253(74)=acd253(3)*acd253(23)
      acd253(74)=acd253(27)+acd253(74)
      acd253(74)=acd253(74)*acd253(70)
      acd253(75)=acd253(56)*acd253(61)
      acd253(76)=-acd253(50)*acd253(60)
      acd253(77)=acd253(48)*acd253(59)
      acd253(72)=acd253(23)*acd253(72)
      acd253(72)=acd253(74)+acd253(72)+acd253(77)+acd253(75)+acd253(76)
      acd253(72)=acd253(1)*acd253(72)
      acd253(74)=acd253(13)*acd253(32)
      acd253(75)=acd253(17)*acd253(74)
      acd253(76)=acd253(15)*acd253(32)
      acd253(77)=acd253(18)*acd253(76)
      acd253(75)=acd253(75)+acd253(77)
      acd253(75)=acd253(11)*acd253(75)
      acd253(74)=acd253(14)*acd253(74)
      acd253(76)=acd253(16)*acd253(76)
      acd253(74)=acd253(74)+acd253(76)
      acd253(74)=acd253(8)*acd253(74)
      acd253(76)=acd253(5)*acd253(23)
      acd253(76)=acd253(29)+acd253(76)
      acd253(70)=acd253(4)*acd253(76)*acd253(70)
      acd253(69)=acd253(69)+acd253(73)+acd253(72)+acd253(71)+acd253(70)+acd253(&
      &75)+acd253(74)
      brack(ninjaidxt1mu0)=acd253(67)
      brack(ninjaidxt0mu0)=acd253(69)
      brack(ninjaidxt0mu2)=acd253(68)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d253h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd253h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k4
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
end module     p2_gg_httbar_d253h0l131_qp
