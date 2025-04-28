module     p2_gg_httbar_d37h8l131
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d37h8l131.f90
   ! generator: buildfortran_tn3.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond_t, d => metric_tensor
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   integer, parameter :: ninjaidxt3mu0 = 0
   integer, parameter :: ninjaidxt2mu0 = 1
   integer, parameter :: ninjaidxt1mu0 = 2
   integer, parameter :: ninjaidxt1mu2 = 3
   integer, parameter :: ninjaidxt0mu0 = 4
   integer, parameter :: ninjaidxt0mu2 = 5
   public :: numerator_t3
contains
!---#[ subroutine brack_31:
   pure subroutine brack_31(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd37h8
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(60) :: acd37
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd37(1)=dotproduct(k2,ninjaE3)
      acd37(2)=abb37(18)
      acd37(3)=dotproduct(l3,ninjaE3)
      acd37(4)=abb37(32)
      acd37(5)=dotproduct(l5,ninjaE3)
      acd37(6)=abb37(50)
      acd37(7)=dotproduct(ninjaE3,spvak1l5)
      acd37(8)=abb37(15)
      acd37(9)=dotproduct(ninjaE3,spvak2e1)
      acd37(10)=abb37(17)
      acd37(11)=dotproduct(ninjaE3,spvae2l3)
      acd37(12)=abb37(19)
      acd37(13)=dotproduct(ninjaE3,spval5l3)
      acd37(14)=abb37(20)
      acd37(15)=dotproduct(ninjaE3,spvak1l3)
      acd37(16)=abb37(21)
      acd37(17)=dotproduct(ninjaE3,spvae2l5)
      acd37(18)=abb37(22)
      acd37(19)=dotproduct(ninjaE3,spvae1l5)
      acd37(20)=abb37(23)
      acd37(21)=dotproduct(ninjaE3,spvak2l3)
      acd37(22)=abb37(24)
      acd37(23)=dotproduct(ninjaE3,spvak2l5)
      acd37(24)=abb37(25)
      acd37(25)=dotproduct(ninjaE3,spval3l5)
      acd37(26)=abb37(27)
      acd37(27)=dotproduct(ninjaE3,spvak2e2)
      acd37(28)=abb37(28)
      acd37(29)=dotproduct(ninjaE3,spval3e2)
      acd37(30)=abb37(29)
      acd37(31)=dotproduct(ninjaE3,spvak2k1)
      acd37(32)=abb37(30)
      acd37(33)=dotproduct(ninjaE3,spvae1l3)
      acd37(34)=abb37(33)
      acd37(35)=dotproduct(ninjaE3,spval3e1)
      acd37(36)=abb37(34)
      acd37(37)=dotproduct(ninjaE3,spval3k2)
      acd37(38)=abb37(36)
      acd37(39)=dotproduct(ninjaE3,spval3k1)
      acd37(40)=abb37(41)
      acd37(41)=acd37(2)*acd37(1)
      acd37(42)=acd37(4)*acd37(3)
      acd37(43)=acd37(6)*acd37(5)
      acd37(44)=acd37(8)*acd37(7)
      acd37(45)=acd37(10)*acd37(9)
      acd37(46)=acd37(12)*acd37(11)
      acd37(47)=acd37(14)*acd37(13)
      acd37(48)=acd37(16)*acd37(15)
      acd37(49)=acd37(18)*acd37(17)
      acd37(50)=acd37(20)*acd37(19)
      acd37(51)=acd37(22)*acd37(21)
      acd37(52)=acd37(24)*acd37(23)
      acd37(53)=acd37(26)*acd37(25)
      acd37(54)=acd37(28)*acd37(27)
      acd37(55)=acd37(30)*acd37(29)
      acd37(56)=acd37(32)*acd37(31)
      acd37(57)=acd37(34)*acd37(33)
      acd37(58)=acd37(36)*acd37(35)
      acd37(59)=acd37(38)*acd37(37)
      acd37(60)=acd37(40)*acd37(39)
      acd37(41)=acd37(60)+acd37(59)+acd37(58)+acd37(57)+acd37(56)+acd37(55)+acd&
      &37(54)+acd37(53)+acd37(52)+acd37(51)+acd37(50)+acd37(49)+acd37(48)+acd37&
      &(47)+acd37(46)+acd37(45)+acd37(44)+acd37(43)+acd37(41)+acd37(42)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd37(41)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d37h8_ninja_t3")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd37h8
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      vecA(1:4) = + a(0:3)
      vecB(1:4) = + b(0:3)
      vecC(1:4) = + c(0:3)
      if (deg.lt.0) return
      t1 = 0
      call cond_t(epspow.eq.t1,brack_31,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p2_gg_httbar_d37h8l131
