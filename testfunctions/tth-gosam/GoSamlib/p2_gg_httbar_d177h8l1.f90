module     p2_gg_httbar_d177h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d177h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd177h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc177(30)
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvae1l5
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      acc177(1)=abb177(12)
      acc177(2)=abb177(13)
      acc177(3)=abb177(14)
      acc177(4)=abb177(15)
      acc177(5)=abb177(16)
      acc177(6)=abb177(17)
      acc177(7)=abb177(19)
      acc177(8)=abb177(20)
      acc177(9)=abb177(21)
      acc177(10)=abb177(23)
      acc177(11)=abb177(25)
      acc177(12)=abb177(26)
      acc177(13)=abb177(27)
      acc177(14)=abb177(29)
      acc177(15)=abb177(30)
      acc177(16)=abb177(33)
      acc177(17)=abb177(39)
      acc177(18)=abb177(43)
      acc177(19)=abb177(44)
      acc177(20)=abb177(55)
      acc177(21)=acc177(2)*Qspvak1e1
      acc177(22)=acc177(10)*Qspvae2e1
      acc177(23)=acc177(12)*Qspvak2e1
      acc177(24)=acc177(14)*Qspval4e1
      acc177(25)=acc177(18)*Qspval5e1
      acc177(21)=acc177(25)+acc177(24)+acc177(23)+acc177(22)+acc177(4)+acc177(2&
      &1)
      acc177(21)=Qspvae1k2*acc177(21)
      acc177(22)=acc177(6)*Qspvae1e2
      acc177(23)=acc177(13)*Qspvae1l4
      acc177(24)=acc177(15)*Qspvae1k1
      acc177(25)=-acc177(17)*Qspvae1l5
      acc177(22)=acc177(25)+acc177(24)+acc177(23)+acc177(22)+acc177(3)
      acc177(22)=Qspval4e1*acc177(22)
      acc177(23)=acc177(5)*Qspvak1e1
      acc177(24)=acc177(7)*Qspvae2e1
      acc177(25)=acc177(8)*Qspvae1k1
      acc177(26)=acc177(9)*Qspvae1e2
      acc177(27)=acc177(11)*Qspvak2e1
      acc177(28)=acc177(16)*Qspvae1l4
      acc177(29)=acc177(19)*Qspvae1l5
      acc177(30)=acc177(20)*Qspval5e1
      brack=acc177(1)+acc177(21)+acc177(22)+acc177(23)+acc177(24)+acc177(25)+ac&
      &c177(26)+acc177(27)+acc177(28)+acc177(29)+acc177(30)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d177h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd177h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d177
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k4
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d177 = 0.0_ki
      d177 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d177, ki), aimag(d177), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d177h8l1
