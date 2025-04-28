module     p2_gg_httbar_d22h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d22h8l1.f90
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
      use p2_gg_httbar_abbrevd22h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc22(29)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: QspQ
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      QspQ = dotproduct(Q,Q)
      acc22(1)=abb22(8)
      acc22(2)=abb22(9)
      acc22(3)=abb22(10)
      acc22(4)=abb22(11)
      acc22(5)=abb22(12)
      acc22(6)=abb22(15)
      acc22(7)=abb22(16)
      acc22(8)=abb22(18)
      acc22(9)=abb22(19)
      acc22(10)=abb22(20)
      acc22(11)=abb22(21)
      acc22(12)=abb22(22)
      acc22(13)=abb22(23)
      acc22(14)=abb22(24)
      acc22(15)=abb22(25)
      acc22(16)=abb22(26)
      acc22(17)=-Qspval4k2*acc22(14)
      acc22(17)=acc22(13)+acc22(17)
      acc22(17)=Qspvak2l5*acc22(17)
      acc22(18)=Qspval4k1*acc22(14)
      acc22(18)=acc22(8)+acc22(18)
      acc22(18)=Qspvak1l5*acc22(18)
      acc22(19)=acc22(5)*Qspvak2k1
      acc22(19)=acc22(19)+acc22(3)
      acc22(19)=Qspvak1k2*acc22(19)
      acc22(20)=acc22(2)*Qspk2
      acc22(21)=acc22(4)*Qspk2**2
      acc22(22)=acc22(7)*Qspval4k2
      acc22(23)=acc22(9)*Qspval4k1
      acc22(24)=acc22(12)*Qspvak2k1
      acc22(25)=Qspval3k2*acc22(10)
      acc22(26)=Qspval3k1*acc22(11)
      acc22(27)=Qspvak2l3*acc22(16)
      acc22(28)=Qspvak1l3*acc22(15)
      acc22(29)=QspQ*acc22(6)
      brack=acc22(1)+acc22(17)+acc22(18)+acc22(19)+acc22(20)+acc22(21)+acc22(22&
      &)+acc22(23)+acc22(24)+acc22(25)+acc22(26)+acc22(27)+acc22(28)+acc22(29)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d22h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd22h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d22
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d22 = 0.0_ki
      d22 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d22, ki), aimag(d22), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d22h8l1
