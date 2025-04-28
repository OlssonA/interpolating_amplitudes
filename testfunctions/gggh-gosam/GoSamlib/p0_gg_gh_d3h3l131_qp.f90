module     p0_gg_gh_d3h3l131_qp
   ! file: /mt/home/sjones/repos/POWHEG-BOX/ggh-gosam/GoSam_POWHEG/Virtual/p0_g &
   ! &g_gh/helicity3d3h3l131_qp.f90
   ! generator: buildfortran_tn3.py
   use p0_gg_gh_config, only: ki => ki_qp
   use p0_gg_gh_util_qp, only: cond_t, d => metric_tensor
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
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(7) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd3(1)=dotproduct(ninjaE3,spvak2k1)
      acd3(2)=dotproduct(ninjaE3,spvak3k2)
      acd3(3)=abb3(6)
      acd3(4)=dotproduct(ninjaE3,spvak2k3)
      acd3(5)=abb3(9)
      acd3(6)=acd3(3)*acd3(1)
      acd3(7)=acd3(5)*acd3(4)
      acd3(6)=acd3(6)+acd3(7)
      acd3(6)=acd3(2)*acd3(6)
      brack(ninjaidxt3mu0)=0.0_ki
      brack(ninjaidxt2mu0)=acd3(6)
   end subroutine brack_31
!---#] subroutine brack_31:
!---#[ subroutine brack_32:
   pure subroutine brack_32(ninjaA, ninjaE3, ninjaE4, ninjaP, brack)
      use p0_gg_gh_model_qp
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_color_qp
      use p0_gg_gh_abbrevd3h3_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: ninjaA, ninjaE3, ninjaE4
      complex(ki), intent(in) :: ninjaP
      complex(ki), dimension(40) :: acd3
      complex(ki), dimension (0:*), intent(inout) :: brack
      acd3(1)=dotproduct(ninjaE3,spvak2k1)
      acd3(2)=dotproduct(ninjaE4,spvak3k2)
      acd3(3)=abb3(6)
      acd3(4)=dotproduct(ninjaE3,spvak3k2)
      acd3(5)=dotproduct(ninjaE4,spvak2k1)
      acd3(6)=dotproduct(ninjaE4,spvak2k3)
      acd3(7)=abb3(9)
      acd3(8)=dotproduct(ninjaE3,spvak2k3)
      acd3(9)=abb3(11)
      acd3(10)=dotproduct(k2,ninjaE3)
      acd3(11)=abb3(16)
      acd3(12)=dotproduct(k3,ninjaE3)
      acd3(13)=abb3(15)
      acd3(14)=dotproduct(ninjaA,ninjaE3)
      acd3(15)=dotproduct(ninjaA,spvak2k1)
      acd3(16)=dotproduct(ninjaA,spvak3k2)
      acd3(17)=dotproduct(ninjaA,spvak2k3)
      acd3(18)=dotproduct(ninjaE3,spval4k2)
      acd3(19)=abb3(7)
      acd3(20)=dotproduct(ninjaE3,spvak3l4)
      acd3(21)=abb3(8)
      acd3(22)=dotproduct(ninjaE3,spvak3k1)
      acd3(23)=abb3(10)
      acd3(24)=dotproduct(k2,ninjaA)
      acd3(25)=dotproduct(k3,ninjaA)
      acd3(26)=dotproduct(ninjaA,ninjaA)
      acd3(27)=dotproduct(ninjaA,spval4k2)
      acd3(28)=dotproduct(ninjaA,spvak3l4)
      acd3(29)=dotproduct(ninjaA,spvak3k1)
      acd3(30)=abb3(13)
      acd3(31)=acd3(1)*acd3(3)
      acd3(32)=acd3(8)*acd3(7)
      acd3(31)=acd3(31)+acd3(32)
      acd3(32)=acd3(31)*acd3(2)
      acd3(33)=acd3(4)*acd3(7)
      acd3(34)=acd3(33)*acd3(6)
      acd3(35)=acd3(4)*acd3(3)
      acd3(36)=acd3(35)*acd3(5)
      acd3(32)=acd3(9)+acd3(32)+acd3(34)+acd3(36)
      acd3(31)=acd3(16)*acd3(31)
      acd3(34)=acd3(15)*acd3(35)
      acd3(33)=acd3(17)*acd3(33)
      acd3(35)=acd3(10)*acd3(11)
      acd3(36)=acd3(12)*acd3(13)
      acd3(37)=acd3(14)*acd3(9)
      acd3(38)=acd3(18)*acd3(19)
      acd3(39)=acd3(20)*acd3(21)
      acd3(40)=acd3(22)*acd3(23)
      acd3(31)=acd3(40)+acd3(39)+acd3(38)+2.0_ki*acd3(37)+acd3(36)+acd3(35)+acd&
      &3(33)+acd3(34)+acd3(31)
      acd3(33)=ninjaP*acd3(32)
      acd3(34)=acd3(15)*acd3(3)
      acd3(35)=acd3(17)*acd3(7)
      acd3(34)=acd3(35)+acd3(34)
      acd3(34)=acd3(16)*acd3(34)
      acd3(35)=acd3(24)*acd3(11)
      acd3(36)=acd3(25)*acd3(13)
      acd3(37)=acd3(26)*acd3(9)
      acd3(38)=acd3(27)*acd3(19)
      acd3(39)=acd3(28)*acd3(21)
      acd3(40)=acd3(29)*acd3(23)
      acd3(33)=acd3(30)+acd3(40)+acd3(39)+acd3(38)+acd3(37)+acd3(36)+acd3(35)+a&
      &cd3(33)+acd3(34)
      brack(ninjaidxt1mu0)=acd3(31)
      brack(ninjaidxt1mu2)=0.0_ki
      brack(ninjaidxt0mu0)=acd3(33)
      brack(ninjaidxt0mu2)=acd3(32)
   end subroutine brack_32
!---#] subroutine brack_32:
!---#[ subroutine numerator_t3:
   subroutine numerator_t3(ncut, a, b, c, param, deg, coeffs) &
   & bind(c, name="p0_gg_gh_d3h3_qp_ninja_t3")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki => ki_nin
      use p0_gg_gh_globalsl1_qp, only: epspow
      use p0_gg_gh_kinematics_qp
      use p0_gg_gh_abbrevd3h3_qp
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
      if (deg.le.(1+(0))) return
      call cond_t(epspow.eq.t1,brack_32,vecA,vecB,vecC,param,coeffs)
   end subroutine numerator_t3
!---#] subroutine numerator_t3:
end module     p0_gg_gh_d3h3l131_qp
