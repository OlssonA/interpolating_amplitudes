module     p2_gg_httbar_d259h0l131_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d259h0l131_qp.f90
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
      use p2_gg_httbar_abbrevd259h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(1) :: acd259
      complex(ki), dimension (0:*), intent(inout) :: brack
      brack(ninjaidxt2mu0)=0.0_ki
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd259h0_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(62) :: acd259
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd259(1)=dotproduct(ninjaE3,spvae1k2)
      acd259(2)=dotproduct(ninjaE3,spval4e2)
      acd259(3)=dotproduct(ninjaE3,spvae2e1)
      acd259(4)=abb259(41)
      acd259(5)=dotproduct(ninjaE3,spvae1e2)
      acd259(6)=dotproduct(ninjaE3,spval5e1)
      acd259(7)=dotproduct(ninjaE3,spvae2k2)
      acd259(8)=abb259(62)
      acd259(9)=dotproduct(ninjaA,ninjaE3)
      acd259(10)=abb259(54)
      acd259(11)=dotproduct(ninjaE3,spvak2e1)
      acd259(12)=abb259(32)
      acd259(13)=abb259(18)
      acd259(14)=dotproduct(ninjaE3,spval5k1)
      acd259(15)=abb259(14)
      acd259(16)=abb259(15)
      acd259(17)=dotproduct(ninjaE3,spval3e1)
      acd259(18)=abb259(57)
      acd259(19)=dotproduct(ninjaE3,spval4k1)
      acd259(20)=abb259(40)
      acd259(21)=dotproduct(ninjaE3,spvae1l3)
      acd259(22)=abb259(37)
      acd259(23)=abb259(33)
      acd259(24)=abb259(53)
      acd259(25)=dotproduct(ninjaA,spvae1k2)
      acd259(26)=dotproduct(ninjaA,spval4e2)
      acd259(27)=dotproduct(ninjaA,spvae2e1)
      acd259(28)=dotproduct(ninjaA,spvae1e2)
      acd259(29)=dotproduct(ninjaA,spval5e1)
      acd259(30)=dotproduct(ninjaA,spvae2k2)
      acd259(31)=abb259(7)
      acd259(32)=abb259(12)
      acd259(33)=abb259(23)
      acd259(34)=abb259(27)
      acd259(35)=abb259(34)
      acd259(36)=abb259(20)
      acd259(37)=abb259(17)
      acd259(38)=abb259(19)
      acd259(39)=abb259(39)
      acd259(40)=abb259(45)
      acd259(41)=abb259(58)
      acd259(42)=abb259(42)
      acd259(43)=abb259(26)
      acd259(44)=abb259(63)
      acd259(45)=dotproduct(ninjaE3,spval4e1)
      acd259(46)=abb259(24)
      acd259(47)=abb259(36)
      acd259(48)=abb259(61)
      acd259(49)=abb259(35)
      acd259(50)=abb259(48)
      acd259(51)=acd259(7)*acd259(6)*acd259(5)*acd259(8)
      acd259(52)=acd259(1)*acd259(2)*acd259(3)*acd259(4)
      acd259(51)=acd259(51)+acd259(52)
      acd259(52)=acd259(19)*acd259(20)
      acd259(53)=acd259(14)*acd259(15)
      acd259(54)=acd259(21)*acd259(22)
      acd259(55)=acd259(17)*acd259(18)
      acd259(56)=acd259(11)*acd259(12)
      acd259(57)=acd259(6)*acd259(23)
      acd259(58)=acd259(2)*acd259(16)
      acd259(59)=acd259(7)*acd259(24)
      acd259(60)=acd259(1)*acd259(13)
      acd259(61)=2.0_ki*acd259(9)
      acd259(62)=acd259(10)*acd259(61)
      acd259(52)=acd259(62)+acd259(60)+acd259(59)+acd259(58)+acd259(57)+acd259(&
      &56)+acd259(55)+acd259(54)+acd259(52)+acd259(53)
      acd259(52)=acd259(52)*acd259(61)
      acd259(53)=acd259(17)*acd259(44)
      acd259(54)=acd259(11)*acd259(35)
      acd259(55)=acd259(8)*acd259(29)
      acd259(55)=acd259(48)+acd259(55)
      acd259(55)=acd259(5)*acd259(55)
      acd259(56)=acd259(8)*acd259(28)
      acd259(56)=acd259(50)+acd259(56)
      acd259(56)=acd259(6)*acd259(56)
      acd259(53)=acd259(56)+acd259(55)+acd259(53)+acd259(54)
      acd259(53)=acd259(7)*acd259(53)
      acd259(54)=acd259(4)*acd259(26)
      acd259(54)=acd259(38)+acd259(54)
      acd259(54)=acd259(3)*acd259(54)
      acd259(55)=acd259(17)*acd259(37)
      acd259(56)=acd259(11)*acd259(31)
      acd259(57)=acd259(4)*acd259(27)
      acd259(57)=acd259(36)+acd259(57)
      acd259(57)=acd259(2)*acd259(57)
      acd259(54)=acd259(57)+acd259(56)+acd259(54)+acd259(55)
      acd259(54)=acd259(1)*acd259(54)
      acd259(55)=acd259(19)*acd259(42)
      acd259(56)=acd259(14)*acd259(39)
      acd259(57)=acd259(21)*acd259(43)
      acd259(55)=acd259(57)+acd259(55)+acd259(56)
      acd259(55)=acd259(17)*acd259(55)
      acd259(56)=acd259(19)*acd259(33)
      acd259(57)=acd259(14)*acd259(32)
      acd259(58)=acd259(21)*acd259(34)
      acd259(56)=acd259(58)+acd259(56)+acd259(57)
      acd259(56)=acd259(11)*acd259(56)
      acd259(57)=acd259(21)*acd259(49)
      acd259(58)=acd259(8)*acd259(30)
      acd259(58)=acd259(47)+acd259(58)
      acd259(58)=acd259(5)*acd259(58)
      acd259(57)=acd259(57)+acd259(58)
      acd259(57)=acd259(6)*acd259(57)
      acd259(58)=acd259(21)*acd259(41)
      acd259(59)=acd259(4)*acd259(25)
      acd259(59)=acd259(40)+acd259(59)
      acd259(59)=acd259(3)*acd259(59)
      acd259(58)=acd259(58)+acd259(59)
      acd259(58)=acd259(2)*acd259(58)
      acd259(59)=acd259(5)*acd259(45)*acd259(46)
      acd259(52)=acd259(52)+acd259(54)+acd259(53)+acd259(58)+acd259(57)+acd259(&
      &59)+acd259(55)+acd259(56)
      brack(ninjaidxt1mu0)=acd259(51)
      brack(ninjaidxt0mu0)=acd259(52)
      brack(ninjaidxt0mu2)=0.0_ki
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p2_gg_httbar_d259h0_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd259h0_qp
      implicit none
      integer(c_int), intent(in) :: ncut, deg
      complex(ki), dimension(0:3), intent(in) :: a, b, c
      complex(ki), intent(in) :: param
      complex(ki), dimension(4) :: vecA, vecB, vecC
      complex(ki), dimension(0:*), intent(out) :: coeffs
      integer :: t1
      complex(ki), dimension(4) :: qshift
      qshift = k2-k3-k5
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
end module     p2_gg_httbar_d259h0l131_qp
